import os

import pandas as pd
import torch

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import LayerNorm
from torch_geometric.nn import GCNConv as GCN, GATConv as GAT, TransformerConv as TCon
from .optimize_prototypes import create_hypersphere

epsilon = 1e-10


def create_activation(name):
    #判断使用哪个激活函数
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name == "elu":
        return nn.ELU()
    elif name is None:
        return nn.Identity()
    else:
        raise NotImplementedError(f"{name} is not implemented.")


def Conv_type(name):
    #判断使用哪个模型
    if name == "gat":
        return GAT, ('add_self_loops', True)
    elif name == "gcn":
        return GCN, ('add_self_loops', True)
    elif name == "tcon":
        return TCon, ('beta', True)
    else:
        raise NotImplementedError(f"{name} is not implemented.")


class SingleModel(torch.nn.Module):
    def __init__(self, activate='elu', drop=0.0, n_head=1, mode_GNN='GAT', hidden_dims=[], mask=0.3, replace=0.0,
                 mask_edge=0.0, C=20):
        super().__init__()
        ## 将 hidden_dims 列表中的每个元素转换为整数类型
        hidden_dims = [int(i) for i in hidden_dims]
        ## 从转换后的 hidden_dims 列表中解包出输入维度、中间维度和输出维度
        [in_dim, mid_dim, out_dim] = hidden_dims
        # 调用 Conv_type 函数，根据 mode_GNN 参数获取对应的卷积层类型以及相关属性和值
        # Conv 是卷积层类，(attr, value) 是该卷积层需要设置的属性和对应的值
        Conv, (attr, value) = Conv_type(mode_GNN)

        ## 初始化第一个卷积层，输入维度为 in_dim，输出维度为 mid_dim，注意力头的数量为 n_head
        self.conv1 = Conv(in_dim, mid_dim, heads=n_head)
        ## 为第一个卷积层设置指定的属性和值
        setattr(self.conv1, attr, value)

        self.conv2 = Conv(n_head * mid_dim, out_dim, heads=n_head)
        setattr(self.conv2, attr, value)

        self.conv3 = Conv(n_head * out_dim, mid_dim, heads=n_head)
        setattr(self.conv3, attr, value)

        self.conv4 = Conv(n_head * mid_dim, in_dim)
        setattr(self.conv4, attr, value)

        #归一化
        self.norm1 = LayerNorm(n_head * mid_dim)
        self.norm2 = LayerNorm(n_head * out_dim)
        self.norm3 = LayerNorm(n_head * mid_dim)
        self.norm4 = LayerNorm(in_dim)

        # # 定义一个可学习的掩码标记，初始值全为 0，形状为 (1, in_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, in_dim))
        # 调用 create_activation 函数，根据 activate 参数创建对应的激活函数实例
        self.activate = create_activation(activate)

        self.drop = drop
        self.mask = mask
        self.replace = min(replace, mask)
        ## 计算掩码标记的比例
        self.mask_token_rate = 1 - self.replace

        #将传入的 mask_edge 参数赋值给类的属性 self.mask_edge。mask_edge 表示边掩码的比例，即在图数据中随机丢弃边的比例。
        self.mask_edge = mask_edge
        self.fc = nn.Linear(n_head * out_dim, int(C))  # For Classification

    def forward(self, x, edge_index, t, idx):
        #含义：通常表示输入的节点特征矩阵。在图神经网络（GNN）的场景下，每一行代表图中的一个节点，每一列代表节点的一个特征维度。例如，若图中有 N 个节点，每个节点有 D 个特征，那么 x 的形状就是 (N, D)。
        #用于描述图中节点之间的连接关系，一般是一个形状为 (2, E) 的张量，其中 E 是图中边的数量。第一行存储边的起始节点索引，第二行存储边的终止节点索引。例如，edge_index[:, i] = [src, dst] 表示从节点 src 到节点 dst 存在一条边。
        #  作用：在图卷积操作中，edge_index 会被用于传递节点之间的信息，帮助模型捕捉图的结构信息。
        mask_nodes, keep_nodes = None, None
        if self.training:#防止过拟合
            #这是一个条件判断语句，self.training 是 torch.nn.Module 类的一个布尔属性，用于表示模型当前是否处于训练模式。只有在训练模式下才会执行下面代码，评估模型下不执行。
            x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(x, self.mask_token)
            ## drop edges
            edge_index = self.drop_deges(edge_index)
            #随机丢弃边数
        #优化能力，进行泛化处理，p是丢弃率，只有在训练模式下才会丢弃，评估模式下原值返回。
        x = F.dropout(x, p=self.drop, training=self.training)

        #这行代码实现了一个图神经网络中常见的前向传播步骤，依次对输入数据进行卷积操作、激活函数处理、归一化操作，最后应用丢弃（Dropout）操作，得到输出特征 h1。整个过程有助于模型学习到更具鲁棒性和泛化能力的特征表示。
        h1 = F.dropout(self.norm1(self.activate(self.conv1(x, edge_index))), p=self.drop, training=self.training)
        #这行代码是图神经网络前向传播过程中的一个步骤，主要完成了对输入特征 h1 进行卷积、激活和归一化操作，得到新的特征表示 h2。这个过程有助于模型进一步学习和提取图数据中的特征信息。
        h2 = self.norm2(self.activate(self.conv2(h1, edge_index)))

        if self.training:
            h2[mask_nodes] = 0

        h3 = F.dropout(self.norm3(self.activate(self.conv3(h2, edge_index))), p=self.drop, training=self.training)
        #从h1到h4,逐步进行卷积以及优化。
        h4 = F.dropout(self.norm4(self.activate(self.conv4(h3, edge_index))), p=self.drop, training=self.training)

        if len(idx) != 0:
            #h2：上一层（通常是第二个卷积层）的输出特征矩阵，形状一般为 (N, D)，其中 N 是节点数量，D 是特征维度
            #idx：包含需要进行分类预测的节点索引的列表。
            #self.fc：全连接层，输入维度为 D，输出维度为分类的类别数量。
            #当 idx 列表不为空时，代码会执行 class_prediction = self.fc(h2[idx])。这意味着只会对 idx 列表中指定索引的节点进行分类预测。例如，若 idx = [1, 3, 5]，则只会对 h2 中索引为 1、3、5 的节点特征使用全连接层 self.fc 进行分类预测。
            #这种方式适用于只需要对图中部分节点进行分类的场景，比如在半监督学习中，可能只有部分节点有标签，我们只需要对这些有标签的节点进行预测和计算损失，以更新模型参数。
            class_prediction = self.fc(h2[idx])
        else:#当 idx 列表为空时，代码会执行 class_prediction = self.fc(h2)。此时会对 h2 中的所有节点进行分类预测，也就是对整个图的所有节点使用全连接层 self.fc 进行分类。
            class_prediction = self.fc(h2)

        #返回多个结果，包括第二个卷积层的输出特征 h2、第四个卷积层的输出特征 h4、
        #被掩码的节点索引 mask_nodes、未被掩码的节点索引 keep_nodes 以及分类预测结果 class_prediction。这些结果可以用于后续的训练、评估或者分析。
        return h2, h4, mask_nodes, keep_nodes, class_prediction

    def encoding_mask_noise(self, x, mask_token):
        #数据增强模块，增强泛化能力。
        #这行代码获取输入特征矩阵 x 的第一维大小，即节点的数量。x 通常是一个形状为 (num_nodes, feature_dim) 的张量，其中 num_nodes 是节点数量，feature_dim 是每个节点的特征维度。
        num_nodes = x.shape[0]
        #首先获取输入特征矩阵 x 中的节点数量。检查 self.mask 是否大于 0，如果是，则对节点进行掩码操作；否则，直接返回原始输入 x 以及相应的索引信息。在掩码操作中，会根据 self.replace 的值决定是否添加噪声，并且将部分节点的特征替换为掩码标记 mask_token
        if self.mask > 0:
            #掩码操作
            #torch.randperm(num_nodes, device=x.device)：生成一个长度为 num_nodes 的随机排列的索引张量 perm，确保随机选择要掩码的节点。device=x.device 表示将生成的张量放置在与 x 相同的设备
            perm = torch.randperm(num_nodes, device=x.device)
            #num_mask_nodes = int(self.mask * num_nodes)：计算需要掩码的节点数量，self.mask 是一个比例值，表示要掩码的节点占总节点数的比例。
            num_mask_nodes = int(self.mask * num_nodes)
            #mask_nodes = perm[: num_mask_nodes]：从随机排列的索引张量 perm 中选取前 num_mask_nodes 个索引，作为要掩码的节点的索引。
            mask_nodes = perm[: num_mask_nodes]
            #选取剩余的索引，作为不需要掩码的节点的索引。
            keep_nodes = perm[num_mask_nodes:]

            #根据 self.replace 决定是否添加噪声，“噪声” 具体指的是用图中随机选取的其他节点的特征来替换被掩码节点的部分特征，以此来模拟数据中的不确定性和干扰。replace是控制掩码中噪声添加的比例。
            if self.replace > 0:
                out_x = x.clone()#复制一份输入特征矩阵 x，避免直接修改原始数据。
                perm_mask = torch.randperm(num_mask_nodes, device=x.device)

                # 计算需要随机替换的节点数量
                num_noise_nodes = int(self.replace * num_mask_nodes)
                # 从 perm_mask 中选取后 num_noise_nodes 个索引，得到需要被替换的噪声节点索引
                noise_nodes = mask_nodes[perm_mask[-num_noise_nodes:]]

                # 生成一个长度为 num_nodes 的随机排列的索引，选取前 num_noise_nodes 个作为替换用的噪声特征的索引
                noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]
                # 将 out_x 中 noise_nodes 对应的节点特征替换为 x 中 noise_to_be_chosen 对应的节点特征
                out_x[noise_nodes] = x[noise_to_be_chosen]

                # 计算需要被置为 0 的掩码标记节点数量
                token_nodes = mask_nodes[perm_mask[: int(self.mask_token_rate * num_mask_nodes)]]
                out_x[token_nodes] = 0.0

            else:#当 self.replace <= 0 时，代码仅将所有掩码节点的特征置为 0。
                out_x = x.clone()
                token_nodes = mask_nodes
                out_x[mask_nodes] = 0.0

            out_x[token_nodes] += mask_token

            return out_x, (mask_nodes, keep_nodes)
        else:
            return x, ([], torch.arange(num_nodes, device=x.device))

    def drop_deges_b(self, edge_index):#drop_deges_b 函数的主要功能是随机丢弃一部分边（edge），从而对输入的边索引 edge_index 进行处理。
        #增加模型的鲁棒性。
        #计算边的数量。edge_index 是一个二维张量，shape 为 [2, num_edge]，[2, num_edge] 表示这个矩阵有2行，num_edge列，列数代表了边数。
        # 其中第一行表示边的起始节点索引，第二行表示边的结束节点索引，edge_index.shape[1] 就是边的数量
        num_edge = edge_index.shape[1]

        # 创建一个长度为 num_edge 的张量 mask，初始值都为 (1 - self.mask_edge)。
        # self.mask_edge 是一个预先设定的概率值，表示要丢弃的边的比例。
        # 例如，如果 self.mask_edge = 0.2，那么 mask 中的每个元素初始值为 0.8
        mask = torch.FloatTensor(torch.ones(num_edge) * (1 - self.mask_edge))

        # 对 mask 张量进行伯努利抽样。伯努利分布是一种离散概率分布，
        # 这里的抽样结果要么是 0（表示丢弃这条边），要么是 1（表示保留这条边）。
        # 抽样的概率就是 mask 张量中每个元素的值，即 1 - self.mask_edge
        mask = torch.bernoulli(mask)

        # 找到 mask 张量中值为 1 的元素的索引。nonzero() 函数返回一个包含非零元素索引的张量，
        # squeeze(1) 函数将这个张量的维度进行压缩，去掉维度为 1 的维度，
        # 最后将结果转换为 long 类型，得到一个一维的索引张量
        mask_idx = mask.nonzero().squeeze(1).long()

        # 根据 mask_idx 对 edge_index 进行索引操作，只保留那些对应索引的边。
        # 最终返回一个新的 edge_index 张量，其中只包含保留下来的边
        return edge_index[:, mask_idx]

        #这个函数通过伯努利抽样的方式，以 self.mask_edge 的概率随机丢弃输入的 edge_index 中的一些边，最终返回一个经过处理后的 edge_index 张量，其中只包含保留下来的边。

    def drop_deges(self, edge_index):
        #mask_edge在singleModel模型建立时设置的参数。
        if self.mask_edge > 0:
            num_edge = edge_index.shape[1]

            perm = torch.randperm(num_edge, device=edge_index.device)
            keep_edges_idx = perm[:int(num_edge * (1 - self.mask_edge))]

            return edge_index[:, keep_edges_idx.long()]
        else:
            return edge_index


class Rotation_matrix(nn.Module):#翻转模型。对输入的嵌入向量进行处理，通过旋转操作和与原型的交互来实现某种分类或特征变换。在处理过程中，会引入一些可学习的参数以及固定的原型，用于计算与分类相关的预测结果。
    def __init__(self, dim, n, C):
        super().__init__()
        ## 将传入的参数 n 赋值给类的属性 self.n，该参数可能用于控制选择原型的数量
        self.n = n

        # 创建一个线性层 rot，输入维度和输出维度都为 dim，并且不使用偏置项
        rot = nn.Linear(dim, dim, bias=False)
        # 对 rot 线性层的权重进行初始化，将其初始化为单位矩阵
        nn.init.eye_(rot.weight)
        # 使用 nn.utils.parametrizations.orthogonal 对 rot 线性层进行正交约束，确保其权重矩阵是正交矩阵
        self.rot = nn.utils.parametrizations.orthogonal(rot)

        # 创建另一个线性层 fc，输入维度和输出维度都为 dim，并且不使用偏置项
        self.fc = nn.Linear(dim, dim, bias=False)
        # 对 fc 线性层的权重进行初始化，将其初始化为单位矩阵
        nn.init.eye_(self.fc.weight)
        # nn.init.xavier_normal_(self.fc.weight)

        # 创建一个 LeakyReLU 激活函数实例，负斜率设置为 0.2
        self.LeakyReLU = nn.LeakyReLU(0.2)

        # 调用 create_hypersphere 函数生成 C 个维度为 dim 的原型向量，并进行归一化处理
        # 将归一化后的原型向量转换为可训练的参数，但设置其 requires_grad 为 False，即这些原型向量在训练过程中不更新
        self.prototypes = nn.Parameter(F.normalize(create_hypersphere(int(C), dim))).requires_grad_(False)

    def forward(self, emb, t):
        '''emb：输入的嵌入向量，通常是节点或样本的特征表示。
t：温度参数，用于控制原型偏移的程度。
a_ij：输入嵌入与原型之间的相似度得分。
idx：相似度得分中每列前 self.n 个最大值的索引。
p：构建的概率分布，用于计算 KL 散度损失。
a_ij_softmax：相似度得分经过 softmax 操作后的概率分布。
kl_loss：KL 散度损失，衡量 p 和 a_ij_softmax 之间的差异。
proto_std：原型的标准差，用于控制原型的偏移量。
eps：随机噪声，用于增加原型的随机性。
proto_shifted：偏移并归一化后的原型。
class_prediction：分类预测结果，是一个概率分布。'''


        #emb：输入的嵌入向量，通常是节点或样本的特征表示。
        # 第一步：计算相似度得分 a_ij
        # self.fc(emb) 对输入嵌入 emb 进行线性变换
        # self.prototypes.t() 是原型矩阵的转置
        # @ 是矩阵乘法运算符
        # 最后通过 LeakyReLU 激活函数引入非线性
        a_ij = self.LeakyReLU(self.fc(emb) @ self.prototypes.t())
        # 第二步：选择前 n 个相似度最高的索引
        # torch.topk 函数返回 a_ij 每列中前 k（即 self.n）个最大值及其索引
        # 这里只取索引 idx，忽略最大值本身
        _, idx = torch.topk(a_ij, dim=0, k=self.n)

        ## 第三步：构建概率分布 p
        # 创建一个与 a_ij 形状相同的全零张量 p
        # requires_grad_(False) 表示该张量不需要计算梯度
        p = torch.zeros_like(a_ij, device=emb.device).requires_grad_(False)
        # # 将 p 中对应 idx 的位置赋值为 1 / self.n
        # torch.arange(p.shape[1]) 生成从 0 到 p 的列数减 1 的整数序列
        p[idx, torch.arange(p.shape[1])] = 1 / self.n

        # 第四步：计算 softmax 后的相似度得分 a_ij_softmax
        # F.softmax 函数对 a_ij 按第 0 维进行 softmax 操作，得到概率分布
        a_ij_softmax = F.softmax(a_ij, dim=0)

        # 第五步：计算 KL 散度损失 kl_loss
        # epsilon 是一个小的常数，用于避免对数运算中的数值不稳定问题
        # (p + epsilon) / (a_ij_softmax + epsilon) 计算两个概率分布的比值
        # torch.log 计算自然对数
        # p * torch.log(...) 逐元素相乘
        # .sum(dim=0, keepdim=True) 按第 0 维求和，并保持维度不变
        kl_loss = (p * torch.log((p + epsilon) / (a_ij_softmax + epsilon))).sum(dim=0, keepdim=True)

        # 第六步：计算原型的标准差 proto_std
        # t 是温度参数，用于控制原型偏移的程度
        # kl_loss.t() 是 kl_loss 的转置
        # torch.ones(1, self.prototypes.shape[1], device=emb.device) 是一个全 1 张量
        # @ 是矩阵乘法运算符
        proto_std = t * kl_loss.t() @ torch.ones(1, self.prototypes.shape[1], device=emb.device)

        # 第七步：生成随机噪声 eps
        # torch.FloatTensor(proto_std.shape).normal_() 生成与 proto_std 形状相同的正态分布随机张量
        # .to(emb.device) 将张量移动到与 emb 相同的设备上
        eps = torch.FloatTensor(proto_std.shape).normal_().to(emb.device)

        # 第八步：对原型进行偏移并归一化得到 proto_shifted
        # self.prototypes + proto_std * eps 对原型进行偏移
        # F.normalize 对偏移后的原型进行归一化处理
        proto_shifted = F.normalize(self.prototypes + proto_std * eps)

        # 第九步：计算分类预测结果 class_prediction
        # self.rot(proto_shifted) 对偏移后的原型进行旋转操作
        # F.normalize 对旋转后的原型进行归一化处理
        # emb @ F.normalize(self.rot(proto_shifted)).t() 计算输入嵌入与旋转归一化后的原型的相似度
        # F.softmax 按最后一维进行 softmax 操作，得到分类概率分布
        class_prediction = F.softmax(emb @ F.normalize(self.rot(proto_shifted)).t(), dim=-1)

        # 返回分类预测结果和偏移后的原型
        return class_prediction, proto_shifted


epsilon = 1e-16


def Loss_recon_graph(G, G_neg, keep_nodes, h2):
    #Loss_recon_graph 函数借助输入的邻接矩阵 G、负样本邻接矩阵 G_neg、未被掩码的节点索引 keep_nodes 以及模型的输出特征 h2，计算图重构损失。该损失的计算一般是基于重构后的图结构和原始图结构的相似度。


    # loss_pos = torch.log(torch.sigmoid((h2[edge_index[0, keep_nodes]]*h2[edge_index[1, keep_nodes]]).sum(dim=-1)) + epsilon)
    # loss_neg = torch.log(1 - torch.sigmoid(h2[edge_index_neg[0, keep_nodes]] @ h2[edge_index_neg[1, keep_nodes]].t()) + epsilon)

    # print(f"keep_nodes: {max(keep_nodes)}")
    # print(f"h2 shape: {h2.shape}")
    # print(f"Indexed h2: {h2[keep_nodes]}")
    # print(f"G shape: {G.shape}")
    # print(f"G_neg shape: {G_neg.shape}")
    # assert keep_nodes.max() < G.size(0), "keep_nodes has indices out of range for G"
    # assert keep_nodes.max() < G_neg.size(0), "keep_nodes has indices out of range for G_neg"

    # 检查 h2[keep_nodes] 的形状是否为二维
    if len(h2[keep_nodes].shape) != 2:
        # 如果 h2[keep_nodes] 不是二维张量，可能需要特殊处理
        # 计算正样本的重构损失
        # G[keep_nodes, :][:, keep_nodes] 提取出只包含 keep_nodes 节点的邻接矩阵子矩阵
        # h2[keep_nodes][0] 取 h2 中 keep_nodes 对应的节点特征的第一个元素
        # h2[keep_nodes][0] @ h2[keep_nodes][0].t() 计算节点特征之间的相似度
        # torch.sigmoid 函数将相似度转换为概率值
        # torch.log 计算对数，epsilon 是一个小的常数，用于避免对数运算中的数值不稳定问题


        loss_pos = G[keep_nodes, :][:, keep_nodes] * torch.log(torch.sigmoid(h2[keep_nodes][0] @ h2[keep_nodes][0].t()) + epsilon)
        # 计算负样本的重构损失
        # G_neg[keep_nodes, :][:, keep_nodes] 提取出只包含 keep_nodes 节点的负样本邻接矩阵子矩阵
        # 1 - torch.sigmoid(...) 计算负样本的概率值
        loss_neg = G_neg[keep_nodes, :][:, keep_nodes] * torch.log(
            1 - torch.sigmoid(h2[keep_nodes][0] @ h2[keep_nodes][0].t()) + epsilon)
    else:
        # 如果 h2[keep_nodes] 是二维张量，进行正常的损失计算
        # 计算正样本的重构损失
        # h2[keep_nodes] @ h2[keep_nodes].t() 计算节点特征之间的相似度矩阵
        loss_pos = G[keep_nodes, :][:, keep_nodes] * torch.log(torch.sigmoid(h2[keep_nodes] @ h2[keep_nodes].t()) + epsilon)
        loss_neg = G_neg[keep_nodes, :][:, keep_nodes] * torch.log(
            1 - torch.sigmoid(h2[keep_nodes] @ h2[keep_nodes].t()) + epsilon)
    # 返回总的图重构损失，负号表示损失是要最小化的目标
    return -loss_pos - loss_neg

'''训练过程中的损失计算和参数更新操作。train_one_epoch 会完成一个完整的训练步骤，包括前向传播、损失计算、反向传播以及参数更新；而 calc_losses 仅进行损失计算，不涉及参数更新。'''
def train_one_epoch(config, fea, h4, h2, keep_nodes, class_prediction, N, G, G_neg, gt, optimizer, scheduler,
                    loss_list):
    #在每次进行反向传播之前，需要将优化器中的梯度信息清零，避免梯度累积影响本次训练。
    optimizer.zero_grad()
    '''config：一个字典，存储训练所需的各种超参数，如 gamma、l1、l2、sched 等，这些参数会影响损失计算和学习率调整。
fea：输入的特征数据，通常是模型的输入特征向量。
h4 和 h2：模型不同层的输出特征，用于后续的损失计算。
keep_nodes：一个索引列表，指示哪些节点的数据会被用于损失计算。
class_prediction：一个字典，包含 'l' 和 'ul' 两个键，分别表示有标签数据和无标签数据的分类预测结果。
N：数据的总数。
G 和 G_neg：图结构相关的数据，可能用于图重构损失的计算。
gt：真实标签，用于计算分类损失。
optimizer：优化器对象，如 torch.optim.Adam，用于更新模型的参数。
scheduler：学习率调度器对象，如 torch.optim.lr_scheduler.StepLR，用于动态调整学习率。
loss_list：一个列表，用于存储每个训练周期的总损失值。'''

    '''首先对 h4 中 keep_nodes 对应的元素进行归一化处理。
然后将归一化后的 h4 与 fea 中 keep_nodes 对应的元素逐元素相乘。
接着对相乘结果在最后一个维度上求和。
用 1 减去求和结果，并将差值进行 gamma 次方运算。
最后对所有元素求平均值，得到重构损失。'''
    loss_recon = ((1 - (F.normalize(h4[keep_nodes]) * fea[keep_nodes]).sum(dim=-1)) ** config[
        'gamma']).mean()
    # loss_discrete = (torch.sqrt(C) - torch.norm(class_prediction['ul'], p=2, dim=0).sum() / torch.sqrt(N2)) / (
    #         torch.sqrt(C) - 1)

    #计算交叉熵损失 loss_ce
    criterion = nn.CrossEntropyLoss()
    # print(class_prediction['l'].shape, gt.shape)
    loss_ce = criterion(class_prediction['l'], gt)

    # print(f"keep_nodes device: {keep_nodes.device}")
    # print(f"G device: {G.device}")
    # print(f"G_neg device: {G_neg.device}")

    #计算图重构损失 loss_recon_graph
    loss_recon_graph = Loss_recon_graph(G, G_neg, keep_nodes, h2).mean()
    #计算总损失 loss
    loss = loss_recon + config['l1'] * loss_ce + config['l2'] * loss_recon_graph
    #反向传播
    loss.backward()
    #参数更新
    optimizer.step()

    #记录损失值
    loss_list.append(loss.detach().cpu().numpy())
    #调整学习率，如果配置文件中的 sched 为 True，则调用学习率调度器的 step 方法调整学习率。
    if config['sched']:
        scheduler.step()
        #返回总损失、重构损失、交叉熵损失和图重构损失的具体数值。
    return loss.detach().item(), loss_recon.detach().item(), loss_ce.detach().item(), loss_recon_graph.detach().item()
'''config：一个字典，其中包含了训练所需的配置参数，像 gamma 和 l2 这样的超参数就在其中。
h2：模型某一层的输出特征，不过在当前函数中并未使用。
h4：模型另一层的输出特征，用于计算重构损失。
keep_nodes：一个索引列表，用来指定在计算损失时要用到的数据节点。
rna：输入的 RNA 数据，通常是特征向量，用于计算重构损失。
class_prediction：模型的分类预测结果，是一个张量，代表模型对输入数据的分类预测。
gt：真实标签，同样是一个张量，代表数据的真实分类。'''
def calc_losses(config, h2, h4, keep_nodes, rna, class_prediction, gt):
    # 计算重构损失
        loss_recon = ((1 - (F.normalize(h4[keep_nodes]) * rna[keep_nodes]).sum(dim=-1)) ** config[
            'gamma']).mean()

        # loss_recon_graph = Loss_recon_graph(G, G_neg, keep_nodes, h2).mean()
         # 创建交叉熵损失函数对象
        criterion = nn.CrossEntropyLoss()
        #计算交叉熵损失
        loss_ce = criterion(class_prediction, gt)

    # 计算总损失，总损失为重构损失加上交叉熵损失乘以权重 l2
        loss = loss_recon + config['l2'] * loss_ce
        # 返回总损失、重构损失和交叉熵损失
        return loss, loss_recon, loss_ce