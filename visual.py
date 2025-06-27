import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.sparse import coo_matrix
import torch

# === 配置参数 ===
DatasetName = 'Nanostring'
ComparisonName = 'rms_gcn'
AblationName = 'train_remote_mask'
fov_list = [15]  # 改为你的 FOV

# 颜色 & 标签映射
color_dict = {
    'lymphocyte': '#E46C6F',
    'neutrophil': '#B268A7',
    'mast': '#6166AD',
    'endothelial': '#3DBDD7',
    'fibroblast': '#25A196',
    'epithelial': '#DCDE49',
    'Mcell': '#FEC918',
    'tumors': '#B3D4E7',
}
label_map = {
    'tumors': 0,
    'fibroblast': 1,
    'lymphocyte': 2,
    'Mcell': 3,
    'neutrophil': 4,
    'endothelial': 5,
    'epithelial': 6,
    'mast': 7,
}
rev_map = {v: k for k, v in label_map.items()}


def plot_graph(save_dir,
               edge_index: np.ndarray,
               gt: np.ndarray,
               spatial_loc: np.ndarray,
               labeled_idx: np.ndarray,
               centers_idx: np.ndarray = None,
               classes_to_plot: list = None,
               bridge_edges: np.ndarray = None):
    os.makedirs(save_dir, exist_ok=True)
    gt_names = np.array([rev_map[int(lbl)] for lbl in gt])
    unique_labels = np.unique(gt_names)
    label_color = {lbl: color_dict[lbl] for lbl in unique_labels}

    if classes_to_plot is None:
        mask_nodes = np.arange(len(gt))
        title = "Total Graph"
    else:
        mask_nodes = np.hstack([np.where(gt_names == cls)[0] for cls in classes_to_plot])
        title = "+".join(classes_to_plot) + " subgraph"

    node_set = set(mask_nodes.tolist())

    bridge_set = set()
    if bridge_edges is not None and bridge_edges.size > 0:
        for u, v in bridge_edges.T:
            bridge_set.add((int(u), int(v)))
            bridge_set.add((int(v), int(u)))

    plt.figure(figsize=(8, 8))

    for u, v in edge_index.T:
        if u in node_set and v in node_set:
            if (u, v) in bridge_set:
                continue
            x0, y0 = spatial_loc[u]
            x1, y1 = spatial_loc[v]
            plt.plot([x0, x1], [y0, y1],
                     color='gray', linewidth=0.5, zorder=1)

    if bridge_set:
        for u, v in bridge_edges.T:
            u, v = int(u), int(v)
            if u in node_set and v in node_set:
                x0, y0 = spatial_loc[u]
                x1, y1 = spatial_loc[v]
                plt.plot([x0, x1], [y0, y1],
                         color='red', linewidth=0.1, linestyle='--', zorder=2)  # 红色线条变细

    for lbl in unique_labels:
        if classes_to_plot is not None and lbl not in classes_to_plot:
            continue
        idx = np.where(gt_names == lbl)[0]
        idx = np.intersect1d(idx, mask_nodes)
        sel = np.intersect1d(idx, labeled_idx)
        unsel = np.setdiff1d(idx, labeled_idx)
        if sel.size > 0:
            plt.scatter(spatial_loc[sel, 0], spatial_loc[sel, 1],
                        c=label_color[lbl], edgecolors='k',
                        s=50, alpha=1.0, zorder=3, label=lbl)
        if unsel.size > 0:
            plt.scatter(spatial_loc[unsel, 0], spatial_loc[unsel, 1],
                        c=label_color[lbl], edgecolors='none',
                        s=50, alpha=0.3, zorder=3)

    if centers_idx is not None:
        for c in centers_idx:
            lbl = rev_map[int(gt[c])]
            if classes_to_plot is None or lbl in classes_to_plot:
                plt.scatter(spatial_loc[c, 0], spatial_loc[c, 1],
                            facecolors='none',
                            edgecolors=label_color[lbl],
                            s=200, linewidths=2, zorder=4)

    plt.title(title)
    plt.axis('off')

    handles = []
    if bridge_set:
        handles.append(plt.Line2D([], [], color='red', linestyle='--',
                                  linewidth=0.1, label='Bridges'))  # 红色线条变细
    if classes_to_plot is None:
        plt.legend(loc='upper right', markerscale=0.6)
    elif handles:
        plt.legend(handles=handles, loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "graph.png"), dpi=300)
    plt.close()


# === 主循环（仅绘图） ===
for fov in fov_list:
    print(f"=== Visualizing FOV {fov} ===")
    fov_path = '.'  # 当前目录

    best_cfg = '.'  # 当前目录
    rd = 'ratio_0.2'  # 手动指定 ratio 文件夹名

    raw_graph_path = 'graph_data.pth'
    data_raw = torch.load(raw_graph_path)
    spatial_loc = data_raw.pos.numpy()
    gt = data_raw.y.numpy()
    labeled_idx = data_raw.train_mask.nonzero(as_tuple=True)[0].cpu().numpy()
    centers_all = data_raw.centers.cpu().numpy() if isinstance(data_raw.centers, torch.Tensor) \
        else np.array(data_raw.centers, dtype=int)

    bridges = data_raw.edge_index_bridges.cpu().numpy()

    edge_rna = data_raw.edge_index_all.numpy()

    # 打印原始图边数
    num_edges_ori = edge_rna.shape[1]
    print(f"Original graph edges: {num_edges_ori}")

    # 这里改成读取权重矩阵，计算平均权重矩阵并提取边索引
    edge_rw_raw = np.load('attn_weights.npy')  # shape (heads, N, N)
    print("edge_rw_raw shape:", edge_rw_raw.shape)

    # 计算平均权重矩阵
    edge_rw_avg = np.mean(edge_rw_raw, axis=0)  # shape (N, N)
    print("edge_rw_avg shape:", edge_rw_avg.shape)

    # 修改部分：按节点保留Top-3边（而非全局Top 1000）
    top_k_per_node = 3
    edge_rw = []
    for i in range(edge_rw_avg.shape[0]):
        top_indices = np.argsort(edge_rw_avg[i])[-top_k_per_node:]  # 每个节点取权重最高的3条边
        for j in top_indices:
            edge_rw.append([i, j])
    edge_rw = np.array(edge_rw).T  # 转换为(2, num_edges)格式
    print(f"Extracted {edge_rw.shape[1]} edges by keeping top {top_k_per_node} edges per node")

    vis_root = os.path.join(best_cfg, 'visualization')
    orig_dir = os.path.join(vis_root, 'original')
    rew_dir = os.path.join(vis_root, 'rewired')
    os.makedirs(orig_dir, exist_ok=True)
    os.makedirs(rew_dir, exist_ok=True)

    plot_graph(orig_dir, edge_rna, gt, spatial_loc, labeled_idx,
               centers_idx=centers_all, classes_to_plot=None,
               bridge_edges=bridges)
    plot_graph(rew_dir, edge_rw, gt, spatial_loc, labeled_idx,
               centers_idx=centers_all, classes_to_plot=None,
               bridge_edges=bridges)

    unique = np.unique([rev_map[int(lbl)] for lbl in gt])
    for cls in unique:
        cls_ori = os.path.join(orig_dir, cls)
        plot_graph(cls_ori, edge_rna, gt, spatial_loc, labeled_idx,
                   centers_idx=centers_all, classes_to_plot=[cls],
                   bridge_edges=bridges)
        cls_rew = os.path.join(rew_dir, cls)
        plot_graph(cls_rew, edge_rw, gt, spatial_loc, labeled_idx,
                   centers_idx=centers_all, classes_to_plot=[cls],
                   bridge_edges=bridges)

    combo = ['tumors', 'mast']
    combo_ori = os.path.join(orig_dir, '_'.join(combo))
    plot_graph(combo_ori, edge_rna, gt, spatial_loc, labeled_idx,
               centers_idx=centers_all, classes_to_plot=combo,
               bridge_edges=bridges)
    combo_rew = os.path.join(rew_dir, '_'.join(combo))
    plot_graph(combo_rew, edge_rw, gt, spatial_loc, labeled_idx,
               centers_idx=centers_all, classes_to_plot=combo,
               bridge_edges=bridges)

print("All done.")
