from Utility.utilities import parameter_setting, seed_torch
from pathlib import Path

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'#指定使用第一个 GPU（如果有多个 GPU 时）。
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'#设置 TensorFlow 输出日志的级别。0 表示显示所有日志
os.environ['R_HOME'] = '/home/wangpenglei/.conda/envs/SC/lib/R'#R_HOME：设置 R 环境的路径。


if __name__ == '__main__':
    # 这是一个字典，用于存储不同数据集（DLPFC、Nanostring、PDAC、human_breast）对应的 margin 值
    margin_dict = {
        'DLPFC': 16,
        'Nanostring': 60,
        'PDAC': 15,
        'human_breast': 10,
    }
    # 这里调用了 parameter_setting 函数（未给出具体实现）来创建一个参数解析器，并使用 parse_args 方法解析命令行参数，将结果存储在 args 中。
    parser = parameter_setting()
    args = parser.parse_args()

    # 根据命令行参数中的 data_path_root 和 dataset 来设置数据路径 args.data_path，并从 margin_dict 中获取对应数据集的 margin 值。
    # args.img_path = args.img_path_root + args.dataset + '/' + args.id +'/1024_256_0.5_512_500/img_emb.npy'
    # args.data_path = args.data_path_root + args.dataset + '/'#路径修改
    args.data_path = r'C:\Users\25684\Desktop\sc-nodes-classification\singlecell-master\SC\Nanostring'

    args.margin = margin_dict[args.dataset]#根据命令行参数中传入的数据集名称（args.dataset），代码从 margin_dict 中提取对应的数据集的 margin 值，并设置数据路径 args.data_path。

    # 这个字典包含了训练模型所需的各种超参数，如学习率（lr）、训练轮数（epoch）、隐藏层维度
    config = {
        'l1': 0.05,#此参数用于调节交叉熵损失（loss_ce）在总损失中的权重，gnns
        'l2': 0.05,#该参数用于调节重构图损失（loss_recon_graph）在总损失中的权重，gnns。
        'hidden_dim': 32,
        'n_head': 1,
        'd_hid': 32,
        'd_emb': 16,
        'lr': 5.0e-03,#学习率，学习率影响步长，可能跳过最优解，一般学习率不宜过大
        'mask': 0.2,#用于对输入数据进行掩码操作，在模型训练时随机掩盖部分数据，以模拟数据缺失或增强模型的鲁棒性。
        'activate': 'elu',#激活函数为模型引入非线性因素，使模型能够学习到数据中的复杂非线性关系。不同的激活函数具有不同的特性，会影响模型的训练效果和性能
        'epoch': 1000,#训练轮数，即模型对整个训练数据集进行训练的次数。较多的训练轮数可能使模型更好地拟合数据，但也可能导致过拟合；较少的训练轮数则可能使模型训练不充分，无法学习到数据的全部特征。
        'drop': 0.0,
        'mask_edge': 0.0,
        'mode': 'TCon',
        'replace': 0.00,#用于指定被掩码数据的替换策略，以一定比例用其他数据或特定值替换被掩码的数据，影响模型对缺失数据的处理方式和学习效果。
        'gamma': 1,
        't': 0.0,
        'sched': True,
        'tolerance': 20,
        'edge_img': False,
        'edge_rna': False
    }
    # 根据命令行参数中的 comparison_name 和 ablation_name 来设置模型和实验结果的保存路径。保存实验结果
    save_path = os.path.join('./Comparison_experiments/' + args.comparison_name, args.ablation_name)#结果保存路径

    # 固定随机种子
    #seed_torch(seed=42)

    from torch.utils.tensorboard import SummaryWriter
    #这部分代码初始化了 TensorBoard 的日志记录器，用于在训练过程中记录日志，以便后续查看训练过程中的各项指标（如损失、准确率等）
    # 创建 SummaryWriter 对象
    model_name = '-'.join([f'{v}' if len(str(v)) < 6 else f'{str(v)[:6]}' for k, v in config.items()])
    model_path = os.path.join(save_path, args.dataset, args.id, model_name,
                              "split_mode_{}_ratio_{}_anc_{}_times_{}".format(args.sm, args.lbr, args.anc, args.times))#lbr是设置的标注数据的比例
    # model_path = os.path.join(save_path, args.dataset, args.id, model_name,
    #                         "alpha_{}".format(args.alpha))
    Path(model_path).mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(model_path)

    # 选择训练模型
    if args.comparison_name == 'graph_transformer':
        from main import train_#修改，调用main函数中的train
    elif args.comparison_name == 'gnns':
        from main_for_gnns import train_
    else:
        raise ValueError('Invalid ablation name')
    model_path, kappa, F1scores, ACC, TPR = train_(config, args=args, model_path=model_path, writer=writer)
    print('End', ACC)#多分类任务（如 6 个类别），则会有 6 个 F1-score，每个值对应一个类别的分类效果。
    # 调用 train_ 函数进行模型训练，并传入配置字典 config、命令行参数 args、模型保存路径 model_path 和 SummaryWriter 对象 writer。训练完成后，获取训练结果（包括模型保存路径、kappa 值、F1 分数、准确率 ACC 和真正率 TPR），并打印准确率 ACC。