from ray import init
import torch
import math
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from scipy.sparse import coo_matrix

from torch_geometric.nn import LayerNorm, GATConv, GCNConv, TransformerConv



class TripletTransformerNetwork(nn.Module):
    '''
    TripletTransformerNetwork is a transformer-based model for dynamic graph structure learning.
    return: predicted nodes labels for labeled nodes, globel attention weights, entropy loss and triplet loss.
    '''
    def __init__(#初始化函数
        self,
        args,
        adj,#邻接矩阵
        n_class,#节点标签的分类数量
        input_dim, #输入维度
        hidden_dim,#隐藏层维度
        output_dim,#输出维度
        nodes_num,#图中节点数量
        n_layers=6,#transformer层数，默认为6层
        num_heads=8,#transformer中多头注意力的头数，默认为8
        dropout = 0.0,
        mask = 0.3,#节点掩码比率，默认为0.0
        mask_edge = 0.3,#边掩码比率，默认为0.3
        replace = 0.0,#替换率，掩码比例中替换的比例
    ):
        super(TripletTransformerNetwork, self).__init__()#保存参数，方便调用
        
        self.adj = adj
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.nodes_num = nodes_num
        self.n_layers = n_layers
        self.num_heads = num_heads
        self.dropout_rate = dropout
        self.ffn_dim = 2 * hidden_dim
        self.margin = 0.5
        self.args = args
        
        self.n_class = n_class

        self.transformer_encoder = nn.ModuleList()#transformer_encoder 是一个 nn.ModuleList，用于存储多个 Transformer 编码器层。

        self.bns = nn.ModuleList()
        self.bns.append(nn.LayerNorm(self.hidden_dim))#self.bns：用于存储 LayerNorm 层，LayerNorm 会对每个样本的特征进行归一化，通常用于稳定训练。
        #LayerNorm，针对 hidden_dim 进行归一化。
        self.fcs = nn.Linear(self.hidden_dim * 2, self.n_class)
        #self.fcs：一个全连接层，将隐藏层的 2 * hidden_dim 特征映射到 n_class 类别空间，用于节点分类任务。
        self.norm1 = LayerNorm(self.num_heads * self.hidden_dim)

        self.norm2 = LayerNorm(self.num_heads * self.output_dim)
        self.norm3 = LayerNorm(self.num_heads * self.hidden_dim)
        #额外的 LayerNorm 层，用于对不同特征维度的归一化，通常会在模型训练过程中帮助稳定训练过程。
        self.norm4 = LayerNorm(self.input_dim)

        # self.activation = F.elu
        self.activate = nn.GELU()

        self.loss = nn.CrossEntropyLoss()#self.loss：交叉熵损失，用于节点分类任务。
        self.triplet_loss = nn.TripletMarginLoss(margin=self.margin)
        #self.triplet_loss：Triplet 损失，用于度量嵌入空间中节点之间的距离。margin=self.margin 是 Triplet 损失的边界，确保同类节点距离较近，不同类节点距离较远。

        # Transformer 卷积层
        self.conv1 = TransformerConv(self.input_dim, self.hidden_dim, heads=self.num_heads)
        #conv1：输入维度到隐藏维度。conv2：隐藏维度到输出维度。conv3：输出维度到隐藏维度。conv4：隐藏维度到输入维度。
        setattr(self.conv1, 'beta', True)
        self.conv2 = TransformerConv(self.num_heads * self.hidden_dim, self.output_dim, heads=self.num_heads)
        setattr(self.conv2, 'beta', True)
        self.conv3 = TransformerConv(self.num_heads * self.output_dim, self.hidden_dim, heads=self.num_heads)
        setattr(self.conv3, 'beta', True)
        self.conv4 = TransformerConv(self.num_heads * self.hidden_dim, self.input_dim)
        setattr(self.conv4, 'beta', True)

        self.mask = mask#mask、replace 和 mask_edge：掩码和替换率，用于图结构学习中的随机遮掩。
        self.replace = replace
        self.mask_edge = mask_edge
        self.mask_token = nn.Parameter(torch.zeros(1, self.input_dim))#mask_token：一个学习的可训练参数，它将被用作图中被遮掩部分的替代表示。

        self.keep_nodes = None#self.keep_nodes 和 self.rna 初始化为 None，
        self.rna = None
        
        self.reset_parameters()

    def reset_parameters(self):#初始化网络层（如 LayerNorm 和 Linear 层）的参数。
        for bn in self.bns:
            bn.reset_parameters()
        self.fcs.reset_parameters()

    def drop_deges(self, edge_index):#该函数用于掩码掉部分边（根据 mask_edge 的比例），随机丢弃图中的一些边。这样可以模拟图中的稀疏性或图的动态变化。
        if self.mask_edge > 0:
            num_edge = edge_index.shape[1]
            #edge_index.shape[1] 返回 edge_index 张量的第二个维度的大小，也就是边的数量 num_edges。
            perm = torch.randperm(num_edge, device=edge_index.device)#生成一个从 0 到 num_edge-1 的随机排列。randperm 会返回一个按随机顺序排列的索引。
            keep_edges_idx = perm[:int(num_edge * (1 - self.mask_edge))]
            #perm[:int(num_edge * (1 - self.mask_edge))]：选择一个数量的边来保留
            #是丢弃边的比例。例如，self.mask_edge = 0.3 时，表示我们丢弃 30% 的边，保留 70% 的边。
            #perm[:700] 选取 perm 中前 700 个随机的索引，表示要保留的边。
            return edge_index[:, keep_edges_idx.long()]
        #edge_index[:, keep_edges_idx.long()]：根据 keep_edges_idx 选择保留的边
        #keep_edges_idx 是一个包含边的索引的张量，这些边会被保留下来。
        else:#如果 self.mask_edge == 0，则不会丢弃任何边，直接返回原始的 edge_index。
            return edge_index
        
    def encoding_mask_noise(self, x, mask_token):#掩码以及噪声处理

        num_nodes = x.shape[0]#通过 x.shape[0] 获取图中的节点数 num_nodes。
        if self.mask > 0:#判断 self.mask 是否大于 0。如果 self.mask > 0，表示我们希望进行掩码操作，掩码的比例为 self.mask。
            perm = torch.randperm(num_nodes, device=x.device)
            #torch.randperm(num_nodes) 生成一个从 0 到 num_nodes-1 的随机排列索引，device=x.device 确保生成的随机数张量在与输入 x 相同的设备上。
            num_mask_nodes = int(self.mask * num_nodes)
            #根据 self.mask 比例计算需要掩码的节点数量。比如，如果 self.mask = 0.3 且 num_nodes = 1000，则 num_mask_nodes = 300。
            mask_nodes = perm[: num_mask_nodes]#掩码节点：选择排列的前 num_mask_nodes 个节点作为需要掩码的节点。
            keep_nodes = perm[num_mask_nodes:]

            if self.replace > 0:
                #判断 self.replace > 0，如果需要进行噪声替换，则会在掩码节点中进行一些替换操作。首先克隆输入 x，得到一个新的张量 out_x，用于存储修改后的节点特征。
                out_x = x.clone()
                #perm_mask：对掩码节点 mask_nodes 进行随机排列，选择要替换的节点。
                perm_mask = torch.randperm(num_mask_nodes, device=x.device)
                #根据 self.replace 比例计算需要替换的噪声节点数量。
                num_noise_nodes = int(self.replace * num_mask_nodes)
                #noise_nodes = mask_nodes[perm_mask[-num_noise_nodes:]]：从 mask_nodes 中选取最后 num_noise_nodes 个节点作为噪声节点。

                noise_nodes = mask_nodes[perm_mask[-num_noise_nodes:]]
                #noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]：生成一个随机排列，选取要替换噪声节点的新特征。
                noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]
                out_x[noise_nodes] = x[noise_to_be_chosen]
                #out_x[noise_nodes] = x[noise_to_be_chosen]：将 noise_nodes 对应的特征替换为从其他节点（x[noise_to_be_chosen]）选取的新特征。
                token_nodes = mask_nodes[perm_mask[: int(self.mask_token_rate * num_mask_nodes)]]
                #token_nodes：从掩码节点中选取一部分（根据 self.mask_token_rate 比例）作为用掩码替代的节点。
                out_x[token_nodes] = 0.0

            else:
                out_x = x.clone()
                token_nodes = mask_nodes
                out_x[mask_nodes] = 0.0

            out_x[token_nodes] += mask_token

            return out_x, (mask_nodes, keep_nodes)
        else:#如果 self.mask == 0，表示不进行掩码操作，直接返回原始的节点特征 x，并且返回空的掩码节点列表和包含所有节点的保留节点索引。
            return x, ([], torch.arange(num_nodes, device=x.device))

    def forward(self, x, adj):
        """
        x: input node features, shape [node_num, input_dim]
        adj: adjacency matrix, shape [node_num, node_num]
        labeled_idx: labeled node indices, shape [labeled_num]
        unlabeled_idx: unlabeled node indices, shape [unlabeled_num]
        """
        edge_index, spatial_loc = adj[0], adj[1]#获取边索引，表示边之间的信息。
        #loc空间位置信息
        self.rna = x#将特征x赋值给self.rna

        mask_nodes, keep_nodes = None, None
        if self.training:
            x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(x, self.mask_token)
            ## drop edges
            edge_index = self.drop_deges(edge_index)
        #进入训练模式时，调用 encoding_mask_noise 函数来注入噪声或进行掩码操作。mask_nodes 是被掩码的节点索引，keep_nodes 是保留的节点索引。
        #然后通过 drop_deges 函数随机丢弃一些边，模拟图的稀疏性。
        self.keep_nodes = keep_nodes

        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        #对节点特征 x 进行丢弃（dropout）操作，p=self.dropout_rate 是丢弃的概率。
        h1 = F.dropout(self.norm1(self.activate(self.conv1(x, edge_index))), 
                       p=self.dropout_rate, training=self.training)
        #经过图卷积层 conv1，激活函数（activate），归一化层（norm1），然后进行 dropout 操作。conv1(x, edge_index) 将输入特征 x 和边的索引 edge_index 进行卷积操作。
        h2, attn = self.conv2(h1, edge_index, return_attention_weights=True)
        #经过第二个图卷积层 conv2，并返回注意力权重 attn，用于后续的可视化或分析。注意力机制通常用于衡量邻接节点的权重。
        h2 = self.norm2(self.activate(h2))
        #对卷积后的结果进行激活（通常是 ReLU 或 GELU）和归一化。
        if self.training:
            h2[mask_nodes] = 0
        #如果是训练模式，将被掩码的节点的特征置为 0，这有助于减少掩码节点对模型的影响。
        h3 = F.dropout(self.norm3(self.activate(self.conv3(h2, edge_index))),
                        p=self.dropout_rate, training=self.training)
        h4 = F.dropout(self.norm4(self.activate(self.conv4(h3, edge_index))), 
                       p=self.dropout_rate, training=self.training)

        #经过所有卷积层后，使用全连接层 fcs 进行节点分类预测。h2 是通过第二层图卷积后的特征，x_out 是最终的分类结果
        x_out = self.fcs(h2)

        return x_out, h4, attn
    
    def loss_calculation(self, h4, x_out, gt, labeled_idx, anchor_idx, positive_idx, negative_idx):
       #计算重建损失（loss_recon），用于衡量图节点的嵌入与原始特征（self.rna）之间的相似度。通过对保留节点（self.keep_nodes）的特征进行归一化（F.normalize），然后计算其与原始特征的点积。

       loss_recon = ((1 - (F.normalize(h4[self.keep_nodes]) * self.rna[self.keep_nodes]).sum(dim=-1))).mean()
       cls_loss = self.loss(x_out[labeled_idx], gt[labeled_idx])
       #计算分类损失（cls_loss），通过交叉熵损失函数 self.loss（通常是 CrossEntropyLoss）来计算模型预测结果 x_out 和真实标签 gt 之间的误差。这里只使用了有标签的节点进行计算。
       loss = self.args.beta * cls_loss + self.args.gamma * loss_recon
       #labeled_idx、anchor_idx、positive_idx、negative_idx 是用于计算损失的索引，分别表示有标签的节点、锚节点、正样本节点和负样本节点。
       return loss
    
