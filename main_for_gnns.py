import os
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from sklearn.metrics.cluster import adjusted_rand_score as ARI, normalized_mutual_info_score as NMI, \
    v_measure_score as VMS
import torch.nn.functional as F
from torch import optim
import pandas as pd
from ray import train
from models.GNNs import SingleModel, calc_losses
import yaml
from scipy.sparse import coo_matrix
from Utility.utilities import metrics, seed_torch, NN_component, load_train_data
from sklearn.neighbors import kneighbors_graph
from tqdm import tqdm

epsilon = 1e-16
#图结构构建
def BuildingInitialGraph(RNA_fea, spatial_loc, gt, knn=7, metric='cosine', add_rna_pos=True):
    # 定义BuildingInitialGraph函数，用于构建图的初始结构
    # 参数：
    # RNA_fea：RNA特征矩阵
    # spatial_loc：空间位置矩阵
    # gt：真实标签
    # knn：K近邻的数量，默认值为7
    # metric：计算距离的度量方式，默认是余弦距离
    # add_rna_pos：是否添加基于RNA特征的正边，默认值为True

    # 基于有标签节点的空间位置的KNN图

    # 基于有标签节点的空间位置的KNN图，构建空间位置的 KNN 图：借助kneighbors_graph函数依据空间位置构建 KNN 图。
    G_loc = kneighbors_graph(np.array(spatial_loc), n_neighbors=knn, mode='connectivity',
                             include_self=False).toarray()
    # 调用kneighbors_graph函数（此函数应来自sklearn.neighbors模块，但代码里未导入），依据空间位置矩阵spatial_loc构建K近邻图
    # n_neighbors=knn：指定K近邻的数量
    # mode='connectivity'：表示图以连接性矩阵的形式返回
    # include_self=False：不包含节点自身作为邻居
    # .toarray()：将稀疏矩阵转换为普通的NumPy数组

    # 基于RNA特征的近邻图

    # 基于RNA特征的近邻图，调用NN_component函数构建近邻图，同时去除自环并避免与空间位置图的边重叠。
    RNA_near, _ = NN_component(RNA_fea, k=knn, metric=metric, mode='and')
    ## 调用NN_component函数（自定义函数），根据RNA特征矩阵RNA_fea构建近邻图
    # k=knn：指定K近邻的数量
    # metric=metric：指定距离度量方式
    # mode='and'：指定构建图的模式
    np.fill_diagonal(RNA_near, 0)  # 去除自环
    RNA_near = np.where(G_loc > 0, 0, RNA_near)  # 避免重叠边

    # 基于RNA特征的负边图
    RNA_far, _ = NN_component(RNA_fea, k=1, mode='or', metric=metric, negative_dis=True)
    # 调用NN_component函数构建基于RNA特征的负边图
    # k=1：指定K近邻数量为1
    # mode='or'：指定构建图的模式
    # negative_dis=True：表示构建负边图
    np.fill_diagonal(RNA_far, 0)  # 去除自环
    RNA_far = np.where(G_loc > 0, 0, RNA_far)  # 避免重叠边


    G_pos = G_loc.copy()
    if add_rna_pos:
        G_pos = np.logical_or(G_loc, RNA_near)  # 合并图
        # 若add_rna_pos为True，则使用np.logical_or函数将G_loc和RNA_near合并成一个新的正边图

    G_neg = RNA_far  # 负边图

    # 创建用于PyTorch Geometric的edge_index
    edge_index = np.vstack((coo_matrix(G_pos).row, coo_matrix(G_pos).col))
    edge_index = torch.from_numpy(edge_index).long()
    # 将NumPy数组转换为torch.Tensor，并将数据类型转换为long

    # 创建负边的edge_index
    edge_index_neg = np.vstack((coo_matrix(G_neg).row, coo_matrix(G_neg).col))
    edge_index_neg = torch.from_numpy(edge_index_neg).long()

    # 转换RNA特征为张量
    RNA_fea = torch.from_numpy(RNA_fea).float()
    #将RNA_fea从NumPy数组转换为torch.Tensor，并将数据类型转换为float

    # 转换图为张量
    G = torch.from_numpy(G_pos)
    # 将正边图G_pos从NumPy数组转换为torch.Tensor
    G_neg = torch.from_numpy(G_neg)
    # 将正边图G_pos从NumPy数组转换为torch.Tensor

    return edge_index, edge_index_neg, RNA_fea, G, G_neg, gt
    # 返回构建好的edge_index、edge_index_neg、RNA_fea、G、G_neg和真实标签gt


def Loss_recon_graph(G, G_neg, keep_nodes, h2):#计算重构图的损失，先计算正边损失，再计算负边损失，最后计算总的损失
    # 定义Loss_recon_graph函数，用于计算图重构损失
    # 参数：
    # G：正边图
    # G_neg：负边图
    # keep_nodes：保留的节点索引
    # h2：模型的中间输出
    loss_pos = G[keep_nodes, :][:, keep_nodes] * torch.log(torch.sigmoid(h2[keep_nodes] @ h2[keep_nodes].t()) + epsilon)
    loss_neg = G_neg[keep_nodes, :][:, keep_nodes] * torch.log(
        1 - torch.sigmoid(h2[keep_nodes] @ h2[keep_nodes].t()) + epsilon)
    # 计算正边的损失
    # G[keep_nodes, :][:, keep_nodes]：提取G中保留节点对应的子图
    # h2[keep_nodes] @ h2[keep_nodes].t()：计算保留节点的特征矩阵的内积
    # torch.sigmoid(...)：对结果应用sigmoid函数
    # torch.log(...) + epsilon：取对数并加上一个小的常数epsilon（代码里未定义），避免对数运算出现数值不稳定的情况
    # 最后将子图和对数结果相乘

    return -loss_pos - loss_neg
 # 返回正边损失和负边损失之和的负值


def train_(config, args=None, model_path=None, writer=None, tune=False):
    # 定义train_函数，是训练的主函数
    # 参数：
    # config：配置字典，包含训练所需的各种参数
    # args：命令行参数，默认为None
    # model_path：模型保存的路径，默认为None
    # writer：用于记录训练过程中的指标，默认为None
    # tune：是否进行超参数调优，默认为False

    with open(os.path.join(model_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
        # 打开指定路径下的config.yaml文件，使用yaml.dump函数将配置字典config以YAML格式写入文件

        # 固定随机种子
        # seed_torch()
        # 注释掉的代码，原本用于固定随机种子，以保证结果的可重复性

    # 固定随机种子
    # seed_torch()

    waiter, min_loss = 0, torch.inf
    # waiter：用于记录损失没有下降的轮数，初始化为0
    # min_loss：记录最小损失，初始化为正无穷
    tolerance = config['tolerance']

    RNA_emb, spatial_loc, gt, labeled_idx, unlabeled_idx, _, _, _ = load_train_data(id=args.id, 
                                                                                knn=args.knn,
                                                                                data_path=args.data_path,
                                                                                margin=args.margin,
                                                                                dataset=args.dataset,
                                                                                labeled_flag='labeled',
                                                                                lbr=args.lbr,
                                                                                anc=args.anc,
                                                                                split_model=args.sm)
    # 调用load_train_data函数加载训练数据
    # 参数从命令行参数args中获取，返回RNA嵌入、空间位置、真实标签、有标签节点索引、无标签节点索引等
     

    ## all data
    edge_index, edge_index_neg, RNA_fea, G, G_neg, gt = BuildingInitialGraph(RNA_emb, spatial_loc, gt, knn=7, 
                                                                             metric='cosine', add_rna_pos= config['edge_rna'])
    
    _, counts = np.unique(gt, return_counts=True)
    counts_min = counts.min()

    edge_index = edge_index.cuda()
    G = G.cuda()
    #用GPU训练
    G_neg = G_neg.cuda()
    gt = torch.tensor(gt)

    num_classes = len(np.unique(gt))
    # 计算真实标签中唯一值的数量，即类别数

    if args.dataset == 'human_breast':
        C = torch.tensor(20, dtype=torch.float).cuda()
        # 若数据集为human_breast，则创建一个值为20的torch.Tensor并移动到GPU上

    rna = F.normalize(RNA_fea.cuda(), dim=-1)
    # 将RNA_fea移动到GPU上，并使用F.normalize函数对其进行归一化处理，归一化维度为最后一维

    assert config['d_emb'] % config['n_head'] == 0
    # 断言config字典中的d_emb能被n_head整除，确保后续模型计算的正确性
    assert config['d_hid'] % config['n_head'] == 0

    loss_list = []
    # loss_list：用于记录每一轮的损失，初始为空列表
    best_test_acc = 0.0
    # best_test_acc：记录最佳的测试准确率，初始化为0

    model = SingleModel(config['activate'], config['drop'], config['n_head'], args.ablation_name,
                        hidden_dims=[rna.shape[1], config['d_hid'] // config['n_head'],
                                     config['d_emb'] // config['n_head']], mask=config['mask'],
                        replace=config['replace'], mask_edge=config['mask_edge'], C=num_classes).cuda()
    # 创建SingleModel模型（自定义模型类），并将其移动到GPU上
    # 参数从配置字典和命令行参数中获取

    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-6)
    # 创建Adam优化器，用于更新模型的参数
    # lr=config['lr']：学习率从配置字典中获取
    # weight_decay=1e-6：权重衰减系数

    scheduler = lambda epoch: (1 + np.cos(epoch / config['epoch'] * 2)) * 0.5
    # 定义一个学习率调度函数，使用余弦退火策略
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
    # 创建LambdaLR学习率调度器，根据调度函数调整学习率


    for epoch in tqdm(range(1, config['epoch']+1), desc='Training'):
        # 开始训练循环，使用tqdm显示进度条
        # range(1, config['epoch']+1)：从1到config['epoch']进行迭代

        model.train()
        optimizer.zero_grad()
        ## semi-supervised train data
        h2, h4, mask_nodes, keep_nodes, class_prediction = model(rna, edge_index, t=config['t'], idx=labeled_idx)
        # 模型进行前向传播，输入为rna、edge_index，t从配置字典中获取，idx为有标签节点的索引
        # 返回中间输出h2、h4，掩码节点索引mask_nodes，保留节点索引keep_nodes和类别预测结果class_prediction
        loss, loss_recon, loss_ce = calc_losses(config, h2, h4, keep_nodes, rna, class_prediction, gt[labeled_idx].cuda())
        # 调用calc_losses函数（自定义函数）计算损失
        # 参数包括配置字典、中间输出、保留节点索引、RNA特征、类别预测结果和有标签节点的真实标签
        # 返回总损失loss、重构损失loss_recon和交叉熵损失loss_ce

  
        loss.backward()#反向传播
        optimizer.step()#参数更新

        if config['sched']:
            scheduler.step()
            # 若配置字典中的sched为True，则调用学习率调度器更新学习率

        if loss < min_loss: # 若当前损失小于最小损失，更新最小损失，将waiter置为0，并保存当前模型的状态
            min_loss = loss
            waiter = 0
            best_model_state = model.state_dict()
        else:
            waiter += 1

        if waiter >= tolerance:
            break
            #跳出训练

        if writer is not None:
            # writer.add_scalar('Accuracy', ACC, epoch)
            writer.add_scalar('loss/loss_recon', loss_recon.detach().cpu().numpy(), epoch)
            writer.add_scalar('loss/loss_ce', loss_ce.detach().cpu().numpy(), epoch)
            writer.add_scalar('loss/loss_sum', loss.detach().cpu().numpy(), epoch)

        ##  Test phase
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                _, _, _, _, class_prediction = model(rna, edge_index, t=config['t'], idx=unlabeled_idx)
                kappa, F1scores, test_acc, TPR= metrics(F.log_softmax(class_prediction, dim=1).argmax(dim=1).cpu().numpy(), 
                                                gt[unlabeled_idx].detach().cpu().numpy(), n_classes=num_classes)
            
            writer.add_scalar('Test_Accuracy', test_acc, epoch)
            writer.add_scalar('Kappa', kappa, epoch)

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_model_state = model.state_dict() 


        if loss.item() < min_loss:
            min_loss = loss.item()
            waiter = 0
        else:
            waiter += 1
            if waiter >= tolerance:
                print("Early stopping triggered.")
                break
            # if tune:
            #     train.report({'ACC': ACC, 'ends_epoch': epoch})

        if waiter >= tolerance:
            model.load_state_dict(best_model_state)

    ## Val Phase
    model.eval()#评估模型，关闭训练时的一些参数
    with torch.no_grad():

        # 上下文管理器，在其作用域内不进行梯度计算，减少内存消耗和计算量
        h2, h4, mask_nodes, keep_nodes, class_prediction = model(rna, edge_index, t=config['t'], idx=[])
        # 模型进行前向传播，输入RNA特征、边索引等，得到中间输出和类别预测
        kappa, F1scores, ACC, TPR= metrics(F.log_softmax(class_prediction, dim=1).argmax(dim=1).cpu().numpy(), 
                                           gt.detach().cpu().numpy(), n_classes=num_classes)
        # 计算评估指标，先对类别预测进行softmax并取最大值得到预测标签，再与真实标签对比计算kappa、F1分数、准确率和每类的召回率
    print(f"Test Accuracy: {ACC}, Kappa: {kappa}, F1scores: {F1scores}")  # 修改 ，打印出结果

    save_model(model_path, model)
    save_outcome(model_path, h2, h4)
    save_metrics(model_path, kappa, F1scores, ACC, TPR)
    # 调用save_metrics函数，将评估指标保存为CSV文件

    if tune: # 如果处于超参数调优模式
        # train.report({'ls':loss, 'Hyper':ari_hyper_final, 'ls_recon':loss_recon, 'ls_dis':loss_discrete, 'ls_graph':loss_recon_graph, 'ends_epoch':epoch})
        train.report({'Accuracy': ACC, 'ends_epoch': epoch})# 使用ray的train模块报告准确率和结束的轮数
    else:
        return model_path, kappa, F1scores, ACC, TPR# 如果不是调优模式，返回模型保存路径和评估指标


def save_outcome(path, h2, h4):
    np.save(file=os.path.join(path, 'latent_space' + '.npy'), arr=h2.detach().cpu().cpu().numpy())
    # 将模型的中间输出h2（潜在空间）从GPU移到CPU，转换为NumPy数组并保存为npy文件
    np.save(file=os.path.join(path, 'recon_rna' + '.npy'), arr=h4.detach().cpu().cpu().numpy())
    # 将模型的中间输出h4（重构的RNA特征）从GPU移到CPU，转换为NumPy数组并保存为npy文件

def save_model(path, model):
    # 保存模型状态字典
    torch.save(model.state_dict(), os.path.join(path, 'model.pt'))
    # 将模型的状态字典保存为pt文件，方便后续加载模型参数

def save_metrics(path, Kappa, F1scores, acc, TPR):
    # 将标量转换为列表
    df = pd.DataFrame({
        'Accuracy': [acc],
        'Kappa': [Kappa * 100],
    }) # 创建一个DataFrame，包含准确率和Kappa系数（乘以100）
    # Add TPR for each class
    for i, tpr in enumerate(F1scores):
        df[f'TPR_Class_{i}'] = tpr * 100 # 为每一类添加召回率（乘以100）到DataFrame中

    df.to_csv(os.path.join(path, 'metrics.csv'), index=False)# 将DataFrame保存为CSV文件，不保存行索引