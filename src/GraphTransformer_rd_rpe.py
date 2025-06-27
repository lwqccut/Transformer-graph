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
        
    def reset_parameters(self):
        for layers in self.transformer_encoder:
            layers.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()

    def forward(self, x, spatial_locs):
        """
        x: input node features, shape [node_num, input_dim]
        adj: adjacency matrix, shape [node_num, node_num]
        labeled_idx: labeled node indices, shape [labeled_num]
        unlabeled_idx: unlabeled node indices, shape [unlabeled_num]
        """

        layer_ = []
        attn_ = []

        z = self.fcs[0](x)
        z = self.bns[0](z)

        z = self.activation(z)
        z = F.dropout(z, p=self.dropout_rate, training=self.training)

        layer_.append(z)

        for i, layer in enumerate(self.transformer_encoder):
            z, attn = layer(z, spatial_locs)
            z += layer_[i]
            z = self.bns[i+1](z)
            z = self.activation(z)
        
            z = F.dropout(z, p=self.dropout_rate, training=self.training)

            layer_.append(z)
            attn_.append(attn)

        # 使用 jumping knowledge 进行信息传递
        # z = torch.cat(layer_, dim=-1)

        x_out = self.fcs[-1](z)

        return x_out, z, attn_
    
    def loss_calculation(self, z, x_out, gt, labeled_idx, anchor_idx, positive_idx, negative_idx):
        anchor = z[anchor_idx]
        positive = z[positive_idx]
        negative = z[negative_idx]

        # compute triplet loss for the current layer
        tri_loss = self.triplet_loss(anchor, positive, negative)
        cls_loss = self.loss(x_out[labeled_idx], gt[labeled_idx])

        loss = self.args.alpha * tri_loss + (1-self.args.alpha) * cls_loss

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

        # self.spd = GraphSPDBias()
        self.rb = GraphRDBias(num_heads, output_size, num_kernel=2)
        self.phi_1 = nn.Parameter(torch.FloatTensor(num_heads, 1, 1), requires_grad=True)
        self.phi_2 = nn.Parameter(torch.FloatTensor(num_heads, 1, 1), requires_grad=True)

        self.reset_parameters()

    def reset_parameters(self):
        self.Wq.reset_parameters()
        self.Wk.reset_parameters()
        self.Wv.reset_parameters()
        self.Wo.reset_parameters()
        torch.nn.init.constant_(self.b, 1.0)

    def forward(self, x, spatial_locs):
        '''
        input:: 
        x: input node features, shape [node_num+1, hidden_size]
        attn_bias: the relative graph attention bias for self-attention, shape [node_num, node_num]
        spatial locs: the spatial locations of nodes，shape (n_nodes, 2)
        return::    
        output: output node features, shape [node_num, hidden_size]
        QK: attention weights, shape [N+1, num_heads, N+1]
        '''

        N = x.size(0)
        
        # attn_spd_bias, _ = self.spd(spatial_locs)
        attn_rd_bias, _ = self.rb(spatial_locs) # [num_heads, N, N]

        # Query, Key, Value
        Query = self.Wq(x).reshape(N, self.num_heads, self.output_size).permute(1, 0, 2) # [num_heads, N, output_size]
        Key = self.Wk(x).reshape(N, self.num_heads, self.output_size).permute(1, 0, 2) # [num_heads, N, output_size]
        Value = self.Wv(x).reshape(N, self.num_heads, self.output_size).permute(1, 0, 2) # [num_heads, N, output_size]

        # Attention
        QK = torch.matmul(Query, Key.transpose(1, 2))  # [num_heads, N, N]
        QK = QK / math.sqrt(self.output_size)

        QK = QK + attn_rd_bias  # [num_heads, N, N] + [num_heads, N, N] -->[num_heads, N, N]

        attention_weights = torch.softmax(QK, dim=-1)  # [num_heads, N, N]
        # attn_bias_2 = torch.mul(attn_rd_bias, self.phi_2)  # [num_heads, N, N] \cdot [num_heads, 1, 1] --> [num_heads, N, N]

        # Apply attention weights to Value
        attended_values = torch.matmul(attention_weights, Value)  # [num_heads, N, output_size]

        # Concatenate多头结果
        attended_values = attended_values.permute(1, 0, 2).reshape(N, -1)  # [N, num_heads * output_size]

        # 最终输出
        output = self.Wo(attended_values)  # [N+1, output_size]

        return output, attention_weights


class EncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.
    """
    def __init__(self, hidden_size, output_size, num_heads = 8):
        super(EncoderLayer, self).__init__()

        self.self_attention = MultiHeadAttention(hidden_size, output_size, num_heads)

    def forward(self, x, spatial_locs):
        """
        input::
        x: input node features, shape [node_num, hidden_size]
        attn_bias: the relative graph attention bias for self-attention, shape [node_num, node_num]
                    1. raw relative graph is regarded as the attention bias at the first layer
                    2. for other layers, the attention bias is computed by the newly generated graphs with the triplet loss
        return::
        output: output node features, shape [node_num, hidden_size]
        """
        x, attn = self.self_attention(x, spatial_locs)

        return x, attn


class GraphRDBias(nn.Module):
    """
    Compute 2D attention bias for a single graph according to the position information for each head.
    """

    def __init__(self, num_heads, embed_dim, num_kernel):
        super(GraphRDBias, self).__init__()
        self.num_heads = num_heads
        self.num_kernel = num_kernel
        self.embed_dim = embed_dim

        # 高斯基函数层，用于对节点的距离进行编码
        self.gbf = GaussianLayer(self.num_kernel)
        
        # 非线性映射层，将高斯基函数结果映射到注意力头的维度
        self.gbf_proj = NonLinear(self.num_kernel, self.num_heads)

        # 如果核的维度和嵌入维度不同，则添加线性投影
        if self.num_kernel != self.embed_dim:
            self.edge_proj = nn.Linear(self.num_kernel, self.embed_dim)
        else:
            self.edge_proj = None

    def forward(self, res_pos):
        """
        :param res_pos: 节点的2D坐标，形状为 (n_nodes, 2)
        :return: 
            graph_attn_bias: 2D attention bias, 形状为 (num_heads, n_nodes, n_nodes)
            merge_edge_features: 投影后的边特征，形状为 (n_nodes, embed_dim)
        """
        n_node = res_pos.shape[0]

        # 计算节点间的距离特征
        dist = res_pos

        # 计算高斯基函数特征
        edge_feature = self.gbf(dist)  # (n_nodes, n_nodes, num_kernel)

        # 将边特征通过非线性层映射到注意力头的维度
        gbf_result = self.gbf_proj(edge_feature)
        graph_attn_bias = gbf_result  # (n_nodes, n_nodes, num_heads)

        # 维度调整为 (num_heads, n_nodes, n_nodes)
        graph_attn_bias = graph_attn_bias.permute(2, 0, 1).contiguous()

        # 投影边特征
        if self.edge_proj is not None:
            sum_edge_features = edge_feature.sum(dim=-2)  # (n_nodes, num_kernel)
            merge_edge_features = self.edge_proj(sum_edge_features)  # (n_nodes, embed_dim)
        else:
            merge_edge_features = edge_feature.sum(dim=-2)  # (n_nodes, num_kernel)

        return graph_attn_bias, merge_edge_features

# 修改后的高斯基函数层，只考虑节点间的二维距离
class GaussianLayer(nn.Module):
    def __init__(self, num_kernel):
        super(GaussianLayer, self).__init__()
        self.num_kernel = num_kernel
        # 用于生成高斯基函数参数的嵌入
        self.mu = nn.Parameter(torch.randn(1, num_kernel))
        self.sigma = nn.Parameter(torch.randn(1, num_kernel))

    def forward(self, dist):
        """
        计算高斯基函数特征
        :param dist: 节点的2D坐标，形状为 (n_nodes, 2)
        :return: 
            edge_feature: 形状为 (n_nodes, n_nodes, num_kernel)
        """
        n_nodes = dist.size(0)
        diff = dist.unsqueeze(1) - dist.unsqueeze(0)  # 计算节点之间的差异
        diff = diff.norm(dim=-1, keepdim=True)  # 计算距离的L2范数

        diff = diff - self.mu
        return torch.exp(-0.5 * (diff ** 2) / (self.sigma ** 2 + 1e-6))

class NonLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(NonLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.linear(x))