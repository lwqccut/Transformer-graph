import torch
import math
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from scipy.sparse import coo_matrix

from torch_geometric.nn import TransformerConv, LayerNorm, GATConv, GCNConv



class TripletTransformerNetwork(nn.Module):
    '''
    TripletTransformerNetwork is a transformer-based model for dynamic graph structure learning.
    return: predicted nodes labels for labeled nodes, globel attention weights, entropy loss and triplet loss.
    '''
    '''args: 可能包含训练过程中其他超参数的配置。
adj: 图的邻接矩阵，表示节点之间的连接关系。
n_class: 预测的类别数，通常是图节点的分类问题。
input_dim: 输入特征的维度。
hidden_dim: Transformer 中间层的隐藏维度。
output_dim: 输出层的维度。
nodes_num: 图中节点的数量。
n_layers: Transformer 编码器的层数（默认 6 层）。
num_heads: Transformer 中的多头注意力头数（默认 8）。
dropout: Dropout 层的比例，防止过拟合。'''
    def __init__(
        self,
        args,
        adj,
        n_class,
        input_dim, 
        hidden_dim,
        output_dim,
        nodes_num,
        n_layers=6,
        num_heads=8,
        dropout = 0.0,
    ):
        super(TripletTransformerNetwork, self).__init__()
        
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

        self.transformer_encoder = nn.ModuleList()

        self.bns = nn.ModuleList()
        self.bns.append(nn.LayerNorm(self.hidden_dim))

        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(input_dim, hidden_dim))
        self.fcs.append(nn.Linear(hidden_dim, self.n_class))

        self.activation = F.elu
        self.loss = nn.CrossEntropyLoss()
        self.triplet_loss = nn.TripletMarginLoss(margin=self.margin)

        for i in range(n_layers):
            self.transformer_encoder.append(
                EncoderLayer(self.hidden_dim, self.hidden_dim, num_heads=num_heads))
            self.bns.append(nn.LayerNorm(self.hidden_dim))
            #在 for 循环中，根据 n_layers 添加多个 Transformer 编码器层
            #每个编码器层包含多头自注意力机制，前馈神经网络，归一化层
        
    def reset_parameters(self):#重新初始化网络中的各个层的参数，确保每个层的权重和偏置都从新初始化的状态开始，避免了训练过程中参数偏移的潜在问题。
        for layers in self.transformer_encoder:
            #调用每个 EncoderLayer 层的 reset_parameters 方法初始化该层的所有参数
            layers.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()

    def forward(self, x, attn_bias):
        """
        x: input node features, shape [node_num, input_dim]
        adj: adjacency matrix, shape [node_num, node_num]
        labeled_idx: labeled node indices, shape [labeled_num]
        unlabeled_idx: unlabeled node indices, shape [unlabeled_num]
        """

        layer_ = []
        #用来存储每一层的输出
        attn_ = []
        #用来存储每一层的注意力权重（attention）

        # position_emds = self.position_embedding(spatial_locs)
        # x += position_emds

        z = self.fcs[0](x)
        #输入特征 x 经过第一个全连接层（self.fcs[0]），将其从 input_dim 维映射到 hidden_dim 维。
        z = self.bns[0](z)
        #经过 Layer Normalization（self.bns[0]）对输出进行归一化，帮助稳定训练过程。

        z = self.activation(z)#使用激活函数（ELU，即 self.activation）进行非线性转换。
        z = F.dropout(z, p=self.dropout_rate, training=self.training)
        #应用 Dropout 正则化，以减少过拟合。p=self.dropout_rate 指定丢弃的比例
        layer_.append(z)#将这一层的输出 z 添加到 layer_ 中，以便后续使用

        for i, layer in enumerate(self.transformer_encoder):
            #通过 enumerate 遍历 self.transformer_encoder 中的每个 Transformer 编码器层。self.transformer_encoder 是一个 ModuleList，包含多个 EncoderLayer，每个编码器层将接受一个节点特征矩阵并输出更新后的节点表示。
            z, attn = layer(z)
            #每一层的编码器将当前层的输入 z 传入，并计算出该层的输出 z 和该层的注意力权重 attn。
            z += layer_[i]
            #通过 跳跃连接（Jumping Knowledge） 将当前层的输出 z 与之前层的输出（存储在 layer_ 中）相加。跳跃连接有助于缓解梯度消失问题，并能够传递来自不同层的信息，增强模型的表达能力。
            z = self.bns[i+1](z)
            #对当前层的输出 z 进行归一化，self.bns[i+1] 是当前层对应的归一化层。通过 LayerNorm，能够加速训练并提高模型的稳定性。
            z = self.activation(z)
            #使用 ELU 激活函数（由 self.activation 定义）对当前层的输出进行非线性变换，增加模型的非线性表示能力。
        
            z = F.dropout(z, p=self.dropout_rate, training=self.training)
            #应用 Dropout 正则化技术，随机丢弃部分神经元，以减少过拟合。self.dropout_rate 控制丢弃的比例，training=self.training 确保在训练时启用 Dropout，而在评估时禁用。

            layer_.append(z)#输出结果存储到layer_
            attn_.append(attn)#注意力权重存储在attn

        # 使用 jumping knowledge 进行信息传递
        # z = torch.cat(layer_, dim=-1)

        x_out = self.fcs[-1](z)#self.fcs 是一个包含多层全连接（Linear）层的 ModuleList，而 self.fcs[-1] 是其中的最后一层。self.fcs[-1] 的输入维度是模型的隐藏维度（hidden_dim），输出维度是 n_class，即每个节点的类别数。
#z：最后一层的节点表示，形状为 [node_num, hidden_dim]，它是 Transformer 编码器经过多层信息传递后的节点嵌入。z 可以作为中间特征，用于进一步的分析、可视化，或在多任务学习中传递给其他任务。
        return x_out, z, attn_#x_out：最终的输出，表示每个节点的分类结果。通常在分类任务中，这些输出会经过 softmax 激活函数（虽然这里没有显式地使用），得到每个节点属于各个类别的概率。返回这个输出作为模型的预测结果。
    
    def loss_calculation(self, z, x_out, gt, labeled_idx, anchor_idx, positive_idx, negative_idx):
        #标签（ground truth），形状为 [node_num]，表示每个节点的真实类别。
        #被标记的节点索引，形状为 [labeled_num]，表示哪些节点有标签，用于计算分类损失。
        anchor = z[anchor_idx]#通过索引 anchor_idx 从 z 中获取锚点的嵌入表示
        positive = z[positive_idx]#通过索引 positive_idx 从 z 中获取正样本的嵌入表示。
        negative = z[negative_idx]#通过索引 negative_idx 从 z 中获取负样本的嵌入表示。

        # compute triplet loss for the current layer
        tri_loss = self.triplet_loss(anchor, positive, negative)
        #这是一个 TripletMarginLoss 损失函数，计算锚点、正样本和负样本之间的距离。三元组损失的目标是最小化锚点和正样本之间的距离，同时最大化锚点和负样本之间的距离。具体来说
        cls_loss = self.loss(x_out[labeled_idx], gt[labeled_idx])
        #这里使用的是 CrossEntropyLoss，它用于分类任务，计算预测类别 x_out[labeled_idx] 和真实标签 gt[labeled_idx] 之间的交叉熵损失。

        loss = self.args.alpha * tri_loss + self.args.beta * cls_loss
        #alpha 控制三元组损失的权重。beta 控制分类损失的权重。
        # loss = cls_loss

        return loss




class MultiHeadAttention(nn.Module):
    '''
    Multi-head attention layer.
    '''
    def __init__(self, hidden_size, output_size, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.output_size = output_size
        
        self.Wq = nn.Linear(hidden_size, num_heads * output_size)
        self.Wk = nn.Linear(hidden_size, num_heads * output_size)
        self.Wv = nn.Linear(hidden_size, num_heads * output_size)
        self.Wo = nn.Linear(num_heads * output_size, output_size)
        self.b = torch.nn.Parameter(torch.FloatTensor(num_heads), requires_grad=True)

    def reset_parameters(self):
        self.Wq.reset_parameters()
        self.Wk.reset_parameters()
        self.Wv.reset_parameters()
        self.Wo.reset_parameters()
        torch.nn.init.constant_(self.b, 1.0)

    def normalize_adj(adj):
        """
        math:
         A_{\text{norm}} = D^{-\frac{1}{2}} \times A \times D^{-\frac{1}{2}} 
        """
        degree = torch.sum(adj, dim=1)
        deg_inv_sqrt = torch.diag(torch.pow(degree, -0.5))
        adj_norm = torch.mm(torch.mm(deg_inv_sqrt, adj), deg_inv_sqrt)
        
        return adj_norm


    def forward(self, x):
        '''
        input:: 
        x: input node features, shape [node_num, hidden_size]
        attn_bias: the relative graph attention bias for self-attention, shape [node_num, node_num]
        return::    
        output: output node features, shape [node_num, hidden_size]
        QK: attention weights, shape [N, num_heads, N]
        '''

        N = x.size(0)

        # Query, Key, Value
        Query = self.Wq(x).reshape(N, self.num_heads, self.output_size).permute(1, 0, 2) # [num_heads, N, output_size]
        Key = self.Wk(x).reshape(N, self.num_heads, self.output_size).permute(1, 0, 2) # [num_heads, N, output_size]
        Value = self.Wv(x).reshape(N, self.num_heads, self.output_size).permute(1, 0, 2) # [num_heads, N, output_size]

        # Attention
        QK = torch.matmul(Query, Key.transpose(1, 2))  # [num_heads, N, N]
        QK = QK / math.sqrt(self.output_size)

        attention_weights = torch.softmax(QK, dim=-1)  # [num_heads, N, N]

        # Apply attention weights to Value
        attended_values = torch.matmul(attention_weights, Value)  # [num_heads, N, output_size]

        # Apply attn bias
        # attn_bias = self.normalization(attn_bias)
        # attn_bias = torch.softmax(attn_bias, dim=-1)
        # attn_bias = attn_bias.unsqueeze(1).repeat(1, self.num_heads, 1).permute(1, 0, 2).float() # [num_heads, N, N]
        # attn_bias = torch.matmul(attn_bias, Value)  # [num_heads, N, output_size]

        # # Add attn bias to attended values
        # attended_values += attn_bias  # [num_heads, N, output_size]

        # Concatenate多头结果
        attended_values = attended_values.permute(1, 0, 2).reshape(N, -1)  # [N, num_heads * output_size]

        # 最终输出
        output = self.Wo(attended_values)  # [N, output_size]

        return output, attention_weights


class EncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.
    """
    def __init__(self, hidden_size, output_size, num_heads = 8):
        super(EncoderLayer, self).__init__()

        self.self_attention = MultiHeadAttention(hidden_size, output_size, num_heads)

    def forward(self, x):
        """
        input::
        x: input node features, shape [node_num, hidden_size]
        attn_bias: the relative graph attention bias for self-attention, shape [node_num, node_num]
                    1. raw relative graph is regarded as the attention bias at the first layer
                    2. for other layers, the attention bias is computed by the newly generated graphs with the triplet loss
        return::
        output: output node features, shape [node_num, hidden_size]
        """

        # compute self-attention on input nodes
        x, attn = self.self_attention(x)

        return x, attn
        #缺少相对位置编码
    
