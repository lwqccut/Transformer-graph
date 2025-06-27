import argparse
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, pairwise_distances
from scipy.sparse import coo_matrix
import random
import os
from sklearn.neighbors import kneighbors_graph
import torch
import scanpy as sc
import torch.nn.functional as F

epsilon = 1e-16

#参数定义函数
def parameter_setting():
    parser = argparse.ArgumentParser()#创建对象

    parser.add_argument('--knn', type=int, default=7, help='Nanostring: 5, DLPFC: 7')#KNN参数
    parser.add_argument('--data_path_root', type=str, default='./SC/')#--data_path_root 表示数据路径的根目录。type=str 表示该参数是字符串类型。default='./SC/' 表示默认值是当前目录下的 ./SC/ 文件夹。
    parser.add_argument('--id', type=str, default='1')#--id 表示一个标识符，通常用于区分不同的视野。默认值是1
    parser.add_argument('--times', type=int, default=0)#--times 表示某些操作执行的次数，默认为 0
    parser.add_argument('--save_path', type=str, default='./SC/GMAE_hyper/Model/')#--save_path 表示模型或结果保存的路径。
    # parser.add_argument('--vis_path', type=str, default='./SC/GMAE_hyper/Model/')#
    parser.add_argument('--device', type=str, default='1')#--device 表示设备 ID，可能是用来指定使用哪个 GPU 或 CPU。默认值是 '1'

    parser.add_argument('--dataset', default='Nanostring', type=str)#--dataset 表示使用的数据集类型，默认是 Nanostring。

    # parser.add_argument('--ablation_name', default='test', type=str)
    parser.add_argument('--ablation_name', default='spatial_loc_rpe_v2', type=str)#--ablation_name 表示进行实验时要使用的 "消融" 配置名称，默认为 spatial_loc_rpe。
    #parser.add_argument('--comparison_name', default='graph_transformer', type=str)
    parser.add_argument('--comparison_name', default='graph_transformer', type=str)#--comparison_name 表示实验中要与哪个方法进行比较，默认为 graph_transformer。
    parser.add_argument('--model_type', type=str, default='TCon')#--model_type 用来选择使用的模型类型，默认为 TCon。
    parser.add_argument('--img_path_root', default='./SC/GMAE_hyper/Img_encoder/models/', type=str)
    #--img_path_root 表示图像模型的路径，默认为 ./SC/GMAE_hyper/Img_encoder/models/。
    parser.add_argument('--lbr', default=0.3, type=float)#--lbr 是一个浮动参数，标注数据的比例，默认为 0.3。
    parser.add_argument('--anc', default=0.15, type=float)#--anc 是另一个浮动参数，默认为 0.15。
    parser.add_argument('--sm', type=str, default='disjoint', help='选择样本剥离的方式')#--sm 是一个字符串类型的参数，用来指定样本剥离的方式。默认为 'disjoint'，帮助信息说明了这个参数是用来选择样本剥离的方式
    parser.add_argument('--alpha', type=float, default=0.4, help='alpha参数')#--alpha 三元组约束的损失权重，0.4为最佳设置
    parser.add_argument('--beta', type=float, default=0.5, help='beta参数')#--beta 消融模型中GraphTransformer_wo_trloss损失计算中分类损失的权重
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma参数')#--gamma 消融模型中GraphTransformer_wo_trloss损失计算中重建损失的权重。
    parser.add_argument('--num_layers', type=int, default=4, help='Transformer的层数')#--num_layers 用来指定 Transformer 网络的层数，默认为 4
    parser.add_argument('--remove_self_loops', type=int, default=1, help='是否移除自环')#--remove_self_loops 是一个整数类型的参数，默认为 1，表示是否移除自环。可能与图模型中的自连接节点有关。
    # 新增保存图数据参数
    parser.add_argument('--save_graph_data', action='store_true', help='保存 graph_data.pt 供可视化使用')


    return parser#返回配置好的 ArgumentParser 对象.


def seed_torch(seed=0):#随机种子
    import torch
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def mclust_R(array, num_cluster, modelNames='EEE', random_seed=None):#array: 输入数据，二维数组（即矩阵），每行代表一个样本，每列代表一个特征。
    import rpy2.robjects as robjects

    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="rpy2")

    robjects.r('suppressMessages(library("mclust"))')
    # robjects.r.library("mclust")
    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()#通过 rpy2.robjects.numpy2ri 实现 Python 的 numpy 数组与 R 语言对象之间的相互转换。这使得 Python 中的 numpy 数组能够直接传递到 R 中。
    if random_seed is not None:
        r_random_seed = robjects.r['set.seed']
        r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']
    # res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(array), num_cluster, modelNames, verbose=False)
    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(array), num_cluster, robjects.NULL, verbose=False)
    mclust_res = np.array(res[-2])
    return mclust_res


def metrics(prediction, target, ignored_labels=[], n_classes=None, target_names=None):
    #计算和输出分类模型的评估指标，包括准确率、混淆矩阵、F1 分数以及每类的正确率
    """Compute and print metrics (accuracy, confusion matrix and F1 scores).
#prediction: 预测标签的列表，模型输出的预测结果。
#target: 真实标签的列表，表示实际的类别。
#ignored_labels: 需要忽略的标签列表（例如某些标签在分析时不重要，可能是未定义的标签）。默认为空列表。
#n_classes: 分类的总类别数，默认为 None，则根据目标标签 target 自动推断类别数。
    Args:
        prediction: list of predicted labels
        target: list of target labels
        ignored_labels (optional): list of labels to ignore, e.g. 0 for undef
        n_classes (optional): number of classes, max(target) by default
    Returns:
        accuracy, F1 score by class, confusion matrix
    """
    ignored_mask = np.zeros(target.shape[:2], dtype=np.bool_)
    #创建一个与目标标签形状相同的布尔数组 ignored_mask，初始值为 False。然后，对于每个要忽略的标签 l，将 target == l 的位置标记为 True，表示这些位置的标签在计算中会被忽略。
    for l in ignored_labels:
        ignored_mask[target == l] = True
    ignored_mask = ~ignored_mask
    #接着，通过 ~ignored_mask 来取反，得到最终的忽略掩码 ignored_mask，这样标记为 True 的位置表示需要计算的标签。

    results = {}

    n_classes = np.max(target) + 1 if n_classes is None else n_classes

    cm = confusion_matrix(
        target,
        prediction,
        labels=range(int(n_classes)))

    classification = classification_report(prediction, target, target_names=target_names)

    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    # 计算每类的正确率
    PerClass_ACC = np.divide((TP + TN), (TP + TN + FP + FN), out=np.zeros_like(TP, dtype=float),
                             where=(TP + TN + FP + FN) != 0)
    results["PerClass_ACC"] = PerClass_ACC

    # Compute global accuracy
    total = np.sum(cm)
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100 / float(total)

    ACC = accuracy

    # Compute F1 score
    F1scores = np.zeros(len(cm))
    for i in range(len(cm)):
        try:
            F1 = 2 * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
        except ZeroDivisionError:
            F1 = 0.
        F1scores[i] = F1

    # Compute kappa coefficient
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / \
         float(total * total)
    kappa = (pa - pe) / (1 - pe)

    return kappa, F1scores, ACC, PerClass_ACC


def NN_component(fea, k=1, metric='cosine', mode='and', negative_dis=False):
    #k指定每个样本的近邻数目，默认为 1，即计算每个样本的最近一个邻居。
    #fea特征矩阵，每行代表一个样本，每列代表一个特征。
    #metric：用于计算样本之间距离的度量方式，默认为 'cosine'（余弦相似度）。常见的还有 'euclidean'（欧氏距离）等。
    #mode：决定合并相邻矩阵时的方式，'and' 表示交集模式（只有两个样本都在彼此的邻居列表中时，才会连接），'or' 表示并集模式（只要一个样本出现在另一个样本的邻居列表中，就会连接）。
    #negative_dis：如果为 True，则在计算距离时取负值，这意味着越远的点会被视为更相似的点。默认为 False。

    if negative_dis:
        dis = -pairwise_distances(fea, metric=metric)
    else:
        dis = pairwise_distances(fea, metric=metric)
        np.fill_diagonal(dis, np.inf)

    idx = np.argsort(dis, axis=-1)
    affinity = np.zeros_like(dis)
    affinity[np.arange(fea.shape[0]).reshape(-1, 1), idx[:, :k]] = 1

    if mode == 'and':
        affinity = np.logical_and(affinity, affinity.T)
    if mode == 'or':
        affinity = np.logical_or(affinity, affinity.T)

    return affinity.astype(float), idx
#这段代码的主要功能是基于 k-近邻算法（k-NN） 构建一个 邻接矩阵，表示样本之间的邻居关系。具体来说，它计算了给定特征矩阵 fea 中每个样本的 k个最近邻，并根据指定的模式（'and' 或 'or'）生成一个二值的邻接矩阵。


def load_train_data(id='151673', knn=7, data_path='/root/GMAE/DLPFC', margin=25, metric='cosine',
                    dim_RNA=3000, dataset='DLPFC', labeled_flag='labeled', lbr=0.3, anc=0.1, split_model='disjoint'):
    from Utility.load_data.load_DLPFC import load_DLPFC_data
    from Utility.load_data.load_Nano import load_Nano_data
    from Utility.load_data.load_PDAC import load_PDAC_data

    if dataset == 'DLPFC' or dataset == 'Human_Breast_Cancer' or dataset == 'Mouse_Brain_Anterior':
        RNA_emb, spatial_loc, gt, adata, labeled_idx, unlabeled_idx = load_DLPFC_data(id=id, path=data_path,
                                                                                      margin=margin,
                                                                                      dim_RNA=dim_RNA,
                                                                                      unlabeled_ratio=lbr,
                                                                                      split_mode=split_model)
    elif dataset == 'Nanostring':
        RNA_emb, gt, spatial_loc, labeled_idx, unlabeled_idx, anchors, positives, negatives = load_Nano_data(int(id),
                                                                                                             margin=margin,
                                                                                                             root_path=data_path,
                                                                                                             labeled_ratio=lbr,
                                                                                                             anchor_ratio=anc,
                                                                                                             split_mode=split_model)

        spatial_loc = np.array(spatial_loc).astype(float)

    elif dataset == 'PDAC':
        patchs, RNA_fea, spatial_loc, gt, _ = load_PDAC_data(path=data_path, margin=margin, dim_RNA=dim_RNA)
    else:
        raise NotImplementedError(f"{dataset} is not implemented.")

    return RNA_emb, spatial_loc, gt, labeled_idx, unlabeled_idx, anchors, positives, negatives


# def load_train_data(id='151673', knn=7, data_path='/root/GMAE/DLPFC', img_path='./', margin=25, metric='cosine',
#                     dim_RNA=3000, dataset='DLPFC', labeled_flag='labeled', lbr=0.3, split_model='disjoint',
#                     add_img_pos=True,
#                     add_rna_pos=True):
#     import torch
#     from .load_data.load_DLPFC import load_DLPFC_data
#     from .load_data.load_Nano import load_Nano_data
#     from .load_data.load_PDAC import load_PDAC_data
#     if dataset == 'DLPFC' or dataset == 'Human_Breast_Cancer' or dataset == 'Mouse_Brain_Anterior':
#         RNA_emb, spatial_loc, gt, adata, labeled_idx, unlabeled_idx = load_DLPFC_data(id=id, path=data_path, margin=margin,
#                                                                        dim_RNA=dim_RNA,
#                                                                        unlabeled_ratio=lbr, split_mode=split_model)
#     elif dataset == 'Nanostring':
#         RNA_emb, patchs, gt, spatial_loc, labeled_idx, unlabeled_idx = load_Nano_data(int(id), margin=margin, root_path=data_path, labeled_ratio=lbr, split_mode=split_model)
#         spatial_loc = np.array(spatial_loc).astype(float)
#     elif dataset == 'PDAC':
#         patchs, RNA_fea, spatial_loc, gt, _ = load_PDAC_data(path=data_path, margin=margin, dim_RNA=dim_RNA)
#     else:
#         raise NotImplementedError(f"{dataset} is not implemented.")

#     return RNA_emb, spatial_loc, gt, labeled_idx, unlabeled_idx


def find_variable_features(adata, nfeat=1500, genes_blocklist='default', min_exp=0.01, max_exp=3):
    sc.pp.highly_variable_genes(adata, n_top_genes=nfeat)#挑选出高变基因，存储再HVG中
    hvg = adata.var[adata.var['highly_variable']].index.tolist()
    #nfeat：选择的高变基因数量，默认为 1500。即选择数据中最具变异性的 nfeat 个基因。
    #genes_blocklist：要排除的基因列表，可以是特定基因的集合。例如，线粒体基因、核糖体基因等。如果是 'default'，会默认排除这些基因。也可以传入一个自定义的基因列表。
    #min_exp：基因的最低平均表达量。默认为 0.01，表示筛选出的基因必须满足平均表达量大于等于 0.01。
    #max_exp：基因的最高平均表达量。默认为 3，表示筛选出的基因必须满足平均表达量小于等于 3。
    if genes_blocklist == 'default':#如果 genes_blocklist 为 'default'，则排除以 MT- 开头（线粒体基因）、RPS 和 RPL（核糖体基因）的基因。
        blocklist = adata.var_names[adata.var_names.str.contains('^MT-|^RPS|^RPL', regex=True)]
        # ToDo: add more and specific blocklist for Nanostring data
    elif genes_blocklist is not None:
        blocklist = genes_blocklist
    else:
        blocklist = []

    hvg = [gene for gene in hvg if gene not in blocklist]

    filtered_hvg = []
    for gene in hvg:
        gene_exp = adata[:, gene].X.toarray().flatten()#提取该基因在所有细胞中的表达数据：adata[:, gene].X.toarray().flatten()。adata[:, gene].X 提取该基因的表达矩阵（稀疏矩阵），然后转换成一维数组。
        avg_exp = np.mean(gene_exp)
        if min_exp <= avg_exp <= max_exp:
            filtered_hvg.append(gene)

    return filtered_hvg


def BuildingInitialGraph(RNA_fea, spatial_loc, gt, labeled_idx, unlabeled_idx, anchors, positives, negatives, knn=7,
                         metric='cosine', num_samples=1):
    """
    构建初始图结构，并选择正样本和负样本，返回三元组的索引，同时保留A, RNA_fea, gt。

    参数:
    - RNA_fea: RNA表达数据
    - spatial_loc: 空间位置信息
    - gt: 标签数据
    - adata: 包含基因表达数据的AnnData对象
    - labeled_idx: 已标注的节点索引
    - knn: 近邻数量
    - metric: 计算分类性能
    - num_samples: 从远端节点中选择的样本数量

    返回:
    - A: 邻接矩阵列表
    - RNA_fea: RNA特征张量
    - gt: 标签数据
    - anchor: 锚点节点的索引
    - positive: 正样本节点的索引
    - negative: 负样本节点的索引
    """
    # 基于空间位置的KNN图（G1），通过 spatial_loc（空间位置信息）构建基于空间邻近的图（邻接矩阵）。
    G_loc = kneighbors_graph(np.array(spatial_loc), n_neighbors=knn, mode='connectivity', include_self=False).toarray()

    # 基于RNA特征的近邻图（G2），基于 RNA 特征 构建一个基于特征的近邻图（邻接矩阵）
    RNA_near, _ = NN_component(RNA_fea, k=knn, metric=metric, mode='and')
    np.fill_diagonal(RNA_near, 0)  # 去除自环

    # 将 anchor-positive 边添加到图中
    for anchor, positive in zip(anchors, positives):
        RNA_near[anchor, positive] = 1.0
        RNA_near[positive, anchor] = 1.0
    #将邻接矩阵 RNA_near 转换为图的 边列表（edge_index）。这通常是图神经网络中用于表示图结构的格式。
    edge_index = np.vstack((coo_matrix(RNA_near).row, coo_matrix(RNA_near).col))

    # 转换为PyTorch Tensor
    # A = torch.from_numpy(RNA_near)
    A = torch.from_numpy(edge_index).long()#将数据转换为 PyTorch Tensor 格式，便于图神经网络的处理。将 edge_index 转换为 PyTorch 的 Long 类型 tensor，表示图的边。
    RNA_fea = torch.from_numpy(RNA_fea).float()
    #将 RNA_fea 转换为 PyTorch 的 Float 类型 tensor，表示细胞的 RNA 特征。

    return A, RNA_fea, gt, anchors, positives, negatives#gt：真实标签或其他地面真相数据，


def count_parameters(model):
    return sum([p.numel() for p in model.parameters() if p.requires_grad])


def dense_to_sparse_tensor(matrix):
    rows, columns = torch.where(matrix > 0)
    values = torch.ones(rows.shape)
    indices = torch.from_numpy(np.vstack((rows,
                                          columns))).long()
    shape = torch.Size(matrix.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def add_zeros(data):
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    return data


def extract_node_feature(data, reduce='add'):
    if reduce in ['mean', 'max', 'add']:
        data.x = torch.scatter(data.edge_attr,
                               data.edge_index[0],
                               dim=0,
                               dim_size=data.num_nodes,
                               reduce=reduce)
    else:
        raise Exception('Unknown Aggregation Type')
    return data


def generate_new_graph(self, anchor_embeds, positive_embeds, negative_embeds, anchor_index, positive_index, num_nodes):
    """
    根据三元组嵌入生成新的图结构，返回一个新的邻接矩阵。

    参数:
    - anchor_embeds: 锚点嵌入
    - positive_embeds: 正样本嵌入
    - negative_embeds: 负样本嵌入
    - anchor_index: 原始图中锚点的节点编号
    """
    num_anchors = anchor_embeds.size(0)#num_anchors：获取锚点的数量，即 anchor_embeds 的第一个维度。size(0) 返回张量的第一个维度的大小，这里即为锚点的数量。

    # 初始化一个新的图矩阵，形状为 (num_nodes, num_nodes)
    num_nodes = int(num_nodes)
    #new_graph：初始化一个大小为 (num_nodes, num_nodes) 的 零矩阵，用来表示新图的邻接矩阵。此矩阵的每个元素将表示图中节点之间的连接权重。
    new_graph = torch.zeros((num_nodes, num_nodes))

    for i in range(num_anchors):
        # 计算当前 anchor 与其对应的 positive 和 negative 的相似度
        positive_sim = F.cosine_similarity(anchor_embeds[i].unsqueeze(0), positive_embeds[i].unsqueeze(0))
        negative_sim = F.cosine_similarity(anchor_embeds[i].unsqueeze(0), negative_embeds[i].unsqueeze(0))

        # 计算连接权重（只考虑正样本的相似度）
        new_value = F.relu(positive_sim - negative_sim).item()

        # 映射 anchor 到原始图的节点编号
        anchor_idx = anchor_index[i]
        positive_idx = positive_index[i]
        new_graph[anchor_idx, positive_idx] = new_value
        new_graph[positive_idx, anchor_idx] = new_value

    # 将新的图矩阵转换为边索引和权重
    edge_index_new = np.vstack((coo_matrix(new_graph).row, coo_matrix(new_graph).col))
    value_new = coo_matrix(new_graph).data
    edge_index_new = torch.from_numpy(edge_index_new).long()
    value_new = torch.from_numpy(value_new).float()

    return edge_index_new, value_new


