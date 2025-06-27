import os
import numpy as np
import torch
from sklearn.metrics.cluster import adjusted_rand_score as ARI, normalized_mutual_info_score as NMI, \
    v_measure_score as VMS
import torch.nn.functional as F
from torch import optim
import pandas as pd
from ray import train
from tqdm import tqdm
from Utility.utilities import metrics, seed_torch, NN_component, load_train_data, BuildingInitialGraph
import yaml
from torch_geometric.data import Data  # 确保在文件顶部 import


def train_(config, args=None, model_path=None, writer=None, tune=False):
    # 保存配置文件,将配置字典 config 以 YAML 格式保存到指定路径的 config.yaml 文件中。
    with open(os.path.join(model_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

    # 固定随机种子
    # seed_torch()
    # waiter 用于记录损失未下降的轮数，min_loss 初始化为正无穷，tolerance 是早停机制的容忍轮数。
    waiter, min_loss = 0, torch.inf
    tolerance = config['tolerance']

    # 调用 load_train_data 函数加载训练数据，再调用 BuildingInitialGraph 函数构建初始图。
    RNA_emb, spatial_loc, gt, labeled_idx, unlabeled_idx, anchors, positives, negatives = load_train_data(id=args.id,
                                                                                                          knn=args.knn,
                                                                                                          data_path=args.data_path,
                                                                                                          margin=args.margin,
                                                                                                          dataset=args.dataset,
                                                                                                          labeled_flag='labeled',
                                                                                                          lbr=args.lbr,
                                                                                                          anc=args.anc,
                                                                                                          split_model=args.sm)

    A, RNA_fea, gt, anchor_idx, positive_idx, negative_idx = BuildingInitialGraph(RNA_emb,
                                                                                  spatial_loc,
                                                                                  gt,
                                                                                  labeled_idx,
                                                                                  unlabeled_idx,
                                                                                  anchors,
                                                                                  positives,
                                                                                  negatives,
                                                                                  knn=7, metric='cosine',
                                                                                  num_samples=3)
    _, counts = np.unique(gt, return_counts=True)  # 统计类别数量，将数据转换为 PyTorch 张量并移动到 GPU 上，对 RNA 特征进行归一化处理。
    num_classes = len(np.unique(gt))#统计独有标签，计算类别
    gt = torch.tensor(gt).cuda()#使用GPU加速训练
    spatial_loc = torch.tensor(spatial_loc).float().cuda()#torch.tensor(spatial_loc): 将 spatial_loc（通常是空间位置信息）转换为 PyTorch 张量。

    num_nodes = torch.tensor(RNA_fea.shape[0], dtype=torch.int)#RNA_fea.shape[0]: 获取 RNA_fea 的行数，即样本数（节点数量）。
    #torch.tensor(..., dtype=torch.int): 转换为 PyTorch int 类型张量，表示 num_nodes（节点数）。

    A = A.cuda()#将 A 从 CPU 移动到 GPU，以加速计算。
    rna = F.normalize(RNA_fea.cuda(), dim=-1)#沿着最后一个维度（即特征维度）进行归一化。

    loss_list = []#是一个 空列表，用于存储训练过程中每个 epoch 的损失值（loss）。
    best_test_acc = 0.0#训练过程中，每个 epoch 计算一次测试集的准确率，如果当前 epoch 的准确率高于 best_test_acc，就更新 best_test_acc，这样最终能保存整个训练过程中的最佳测试准确率。
    #0.0为初始值，有效的正确率一定大于0

    # select the ablation models.选择消融模型
    if args.ablation_name == 'spatial_loc_rpe':#带有空间相对位置的图转换器
        # Y^h = [softmax(XW_Q^h(XW_K^h)^T + \phi_2^h(D))]XW_V^h
        from models.GraphTransformer_rd_rpe import TripletTransformerNetwork

        model = TripletTransformerNetwork(
            args,
            A,
            n_class=num_classes,
            input_dim=rna.shape[1],
            hidden_dim=config['hidden_dim'],
            output_dim=num_classes,
            nodes_num=num_nodes,
            n_layers=6,
            num_heads=8,
            dropout=0.0).cuda()
        spatial_loc_rpe = spatial_loc

        print(
            "Using graph transformer with triplet loss and spatial location relative position encoding, regarded as the completed model")

    elif args.ablation_name == 'wo_triplet_loss':#不适用三元组损失的图转换器
        from models.GraphTransformer_wo_trloss import TripletTransformerNetwork#模块修改,模块二次修改

        model = TripletTransformerNetwork(
            args,
            A,
            n_class=num_classes,
            input_dim=rna.shape[1],
            hidden_dim=config['hidden_dim'],
            output_dim=num_classes,
            nodes_num=num_nodes,
            n_layers=6,
            num_heads=8,
            dropout=0.0).cuda()
        spatial_loc_rpe = [A, spatial_loc]

        print("Using graph transformer without triplet loss and relative position encoding")

    elif args.ablation_name == 'wo_rpe':
        # Y^h = [softmax(XW_Q^h(XW_K^h)^T]XW_V^h
        from models.GraphTransformer import TripletTransformerNetwork

        model = TripletTransformerNetwork(
            args,
            A,
            n_class=num_classes,
            input_dim=rna.shape[1],
            hidden_dim=config['hidden_dim'],
            output_dim=num_classes,
            nodes_num=num_nodes,
            n_layers=6,
            num_heads=8,
            dropout=0.0).cuda()
        spatial_loc_rpe = None

        print("Using normal graph transformer without relative position encoding")
    elif args.ablation_name == 'spatial_loc_rpe_v2':
        # Y^h = [\phi(D) \cdot softmax(XW_Q^h(XW_K^h)^T + \phi_2^h(D))]XW_V^h
        # which D means the relative position, which is made by turning spatial_loc as resistance distance matrix
        from models.GraphTransformer_rd_rpe_v2_GCN import TripletTransformerNetwork

        model = TripletTransformerNetwork(
            args,
            A,
            n_class=num_classes,
            input_dim=rna.shape[1],
            hidden_dim=config['hidden_dim'],
            output_dim=num_classes,
            nodes_num=num_nodes,
            n_layers=6,
            num_heads=8,
            dropout=0.0).cuda()
        spatial_loc_rpe = spatial_loc

        print("Using Spatial location relative position version 2 encoding")

    elif args.ablation_name == 'hyperparameters':
        # Y^h = [softmax(XW_Q^h(XW_K^h)^T + \phi_2^h(D))]XW_V^h
        from models.GraphTransformer_rd_rpe import TripletTransformerNetwork

        model = TripletTransformerNetwork(
            args,
            A,
            n_class=num_classes,
            input_dim=rna.shape[1],
            hidden_dim=config['hidden_dim'],
            output_dim=num_classes,
            nodes_num=num_nodes,
            n_layers=6,
            num_heads=8,
            dropout=0.0).cuda()
        spatial_loc_rpe = spatial_loc

        print(
            "Using graph transformer with triplet loss and spatial location relative position encoding, regarded as the completed model")

    elif args.ablation_name == 'hvg_anchors':
        # Y^h = [softmax(XW_Q^h(XW_K^h)^T + \phi_2^h(D))]XW_V^h
        from models.GraphTransformer_rd_rpe import TripletTransformerNetwork

        model = TripletTransformerNetwork(
            args,
            A,
            n_class=num_classes,
            input_dim=rna.shape[1],
            hidden_dim=config['hidden_dim'],
            output_dim=num_classes,
            nodes_num=num_nodes,
            n_layers=6,
            num_heads=8,
            dropout=0.0).cuda()
        spatial_loc_rpe = spatial_loc

        print(
            "Using graph transformer with triplet loss and spatial location relative position encoding, hvg anchors, regarded as the completed model")
    else:
        raise NotImplementedError(f"{args.ablation_name} is not implemented.")

    # 定义优化器和学习率调度器，使用 Adam 优化器，学习率 lr 从配置中读取，并添加 L2 正则化。
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-6)

    scheduler = lambda epoch: (1 + np.cos(epoch / config['epoch'] * 2)) * 0.5
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)

    # for epoch in range(1, config['epoch'] + 1):在每个训练轮次中，模型处于训练模式，前向传播得到输出，计算损失，反向传播更新参数。
    for epoch in tqdm(range(1, config['epoch'] + 1), desc='Training'):
        model.train()
        x_out, z, attn = model(rna, spatial_loc_rpe)

        loss = model.loss_calculation(z, x_out, gt, labeled_idx, anchor_idx, positive_idx, negative_idx)

        optimizer.zero_grad()#清除梯度
        loss.backward()#反向传播
        optimizer.step()#更新参数

        ##  Test phase,每 10 个轮次进行一次验证，记录验证指标到 TensorBoard，若验证准确率高于之前的最佳值，则保存当前模型状态。
        if epoch % 10 == 0:
            model.eval()
            # Valid
            with torch.no_grad():

                x_out, z, attn = model(rna, spatial_loc_rpe)

                val_loss = model.loss_calculation(z, x_out, gt, unlabeled_idx, anchor_idx, positive_idx, negative_idx)

                kappa, F1scores, test_acc, per_class_acc = metrics(
                    F.log_softmax(x_out, dim=1).argmax(dim=1).cpu().numpy(),
                    gt.detach().cpu().numpy(), n_classes=num_classes)

            writer.add_scalar('accuracy', test_acc, epoch)#准确率
            writer.add_scalar('Kappa', kappa, epoch)#kappa
            writer.add_scalar('tr_loss', loss, epoch)#f1score
            writer.add_scalar('val_loss', val_loss, epoch)#验证损失
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)#学习率

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_model_state = model.state_dict()
                # 早停机制，若当前损失小于最小损失，更新最小损失并重置 waiter；否则 waiter 加 1，当 waiter 超过容忍轮数时触发早停，加载最佳模型状态。
        if loss.item() < min_loss:
            min_loss = loss.item()
            waiter = 0
        else:
            waiter += 1
            if waiter >= tolerance:
                print("Early stopping triggered.")
                break

        if waiter >= tolerance:
            model.load_state_dict(best_model_state)
    # 模型处于评估模式，进行最终评估，打印评估指标，保存模型、指标和输出结果。
    model.eval()
    with torch.no_grad():

        x_out, z, attn = model(rna, spatial_loc_rpe)

        kappa, F1scores, ACC, per_class_acc = metrics(F.log_softmax(x_out, dim=1).argmax(dim=1).cpu().numpy(),
                                                      gt.detach().cpu().numpy(), n_classes=num_classes)

    print(f"Test Accuracy: {ACC}, Kappa: {kappa}, F1scores: {F1scores}")


    save_model(model_path, model)
    save_metrics(model_path, kappa, F1scores, ACC, per_class_acc)
    save_outcome(model_path, attn, z)
    save_graph_data(model_path, spatial_loc, gt, labeled_idx, A)

    # 若 tune 为 True，使用 train.report 报告指标；否则返回模型路径、指标等信息。
    if tune:
        train.report({
            'F1_test': F1scores,
            'Kappa_test': kappa,
            'Accuracy_test': ACC
        })
    else:
        return model_path, kappa, F1scores, ACC, per_class_acc


def save_model(path, model):
    # 保存模型状态字典
    torch.save(model.state_dict(), os.path.join(path, 'model.pt'))


def save_outcome(path, attn_weights, layers_z):
    # 保存each layer注意力权重
    # for i, attn in enumerate(attn_weights):
    attn = attn_weights[-1]
    np.save(file=os.path.join(path, f'attn_weights.npy'), arr=attn.detach().cpu().numpy())
    # 保存隐藏层输出
    np.save(file=os.path.join(path, f'features_embedding.npy'), arr=layers_z.detach().cpu().numpy())


def save_metrics(path, Kappa, F1scores, acc, TPR):
    # 将标量转换为列表
    df = pd.DataFrame({
        'Accuracy': [acc],
        'Kappa': [Kappa * 100],
    })

    # Add TPR for each class
    for i, tpr in enumerate(F1scores):
        df[f'TPR_Class_{i}'] = tpr * 100

    df.to_csv(os.path.join(path, 'metrics.csv'), index=False)


# def save_graph_data(path, spatial_loc, gt, labeled_idx, A):
#     """
#     保存图数据到指定路径，供后续可视化使用。
#
#     参数:
#     - path: 保存目录（字符串）
#     - spatial_loc: 节点空间位置，形状 [N, 2]（Tensor）
#     - gt: 节点标签，形状 [N]（Tensor）
#     - labeled_idx: 已标注节点索引（1D Tensor）
#     - A: 稠密邻接图（torch.sparse 矩阵）
#
#     保存文件:
#     - path/graph_data.pt
#     """
#     from torch_geometric.data import Data
#     import torch
#     import os
#
#     edge_index_all = A._indices().cpu()
#     data = Data(
#         pos=spatial_loc.detach().cpu(),
#         y=gt.detach().cpu(),
#         train_mask=torch.zeros_like(gt, dtype=torch.bool).scatter(0, labeled_idx, True),
#         centers=labeled_idx.cpu(),
#         edge_index_all=edge_index_all,
#         edge_index_bridges=edge_index_all  # 暂时复用；如有 bridge 信息可替换
#     )
#     torch.save(data, os.path.join(path, 'graph_data.pt'))
#     print(f"[✓] 图数据已保存至: {os.path.join(path, 'graph_data.pt')}")
def save_graph_data(path, spatial_loc, gt, labeled_idx, edge_index):
    """
    保存图数据为 .pth 文件，供可视化使用（确保是 KNN 图结构）。

    参数:
    - path: 保存目录
    - spatial_loc: [N, 2] 节点空间位置（Tensor）
    - gt: [N] 节点标签（Tensor）
    - labeled_idx: 已标注节点索引（list 或 Tensor）
    - edge_index: [2, E] 的 edge_index 图结构
    """
    import os
    import torch
    from torch_geometric.data import Data

    if not isinstance(edge_index, torch.Tensor):
        edge_index = torch.tensor(edge_index, dtype=torch.long)

    if not isinstance(labeled_idx, torch.Tensor):
        labeled_idx = torch.tensor(labeled_idx, dtype=torch.long)

    train_mask = torch.zeros_like(gt, dtype=torch.bool)
    train_mask[labeled_idx] = True

    data = Data(
        pos=spatial_loc.detach().cpu(),
        y=gt.detach().cpu(),
        train_mask=train_mask.cpu(),
        centers=labeled_idx.cpu(),
        edge_index_all=edge_index.cpu(),
        edge_index_bridges=edge_index.cpu()
    )

    os.makedirs(path, exist_ok=True)
    save_path = os.path.join(path, 'graph_data.pth')
    torch.save(data, save_path)
    print(f"[✓] 图数据（KNN结构）已保存至: {save_path}")
