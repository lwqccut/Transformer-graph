import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import cv2
import random
import os
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans


def split_labeled_data(spatial_loc, labels, labeled_ratio=0.3, seed=42, split_mode='disjoint'):
    """
    Split the dataset into labeled and unlabeled indices using the specified method.

    Parameters
    ----------
    spatial_loc : ndarray
        The spatial locations for each sample.
    labels : ndarray
        The labels for the entire dataset.
    labeled_ratio : float, optional
        Ratio of samples to retain labels.
    seed : int, optional
        Random seed for reproducibility.
    split_mode : str, optional
        The splitting mode.

    Returns
    -------
    labeled_idx : list
        Indices of labeled samples.
    unlabeled_idx : list
        Indices of unlabeled samples.
    """
    np.random.seed(seed)
    random.seed(seed)

    # spatial_loc
    # [3276.  9178.  5133.  3462.  2779.  3053.  5109.  8830. 10228. 10075...]
    # [2514.  8520.  2878.  9581.  7663.  8143. 11263.  9837.  2894.  7924...]

    # Create the ground truth (gt) matrix
    spatial_loc = spatial_loc.astype(int)
    labels = labels + 1
    x_max, y_max = spatial_loc[:, 0].max() + 1, spatial_loc[:, 1].max() + 1
    gt = -1 * np.ones((x_max, y_max), dtype=int)
    for idx, xy in enumerate(spatial_loc):
        gt[xy[0], xy[1]] = labels[idx]

    unique_labels = np.unique(labels)

    # Save data indices
    labeled_idx, unlabeled_idx = [], []

    if split_mode == 'random':
        for label in unique_labels:
            X = np.where(gt == label)
            X = list(zip(*X))

            train_gt = np.zeros_like(gt)
            test_gt = np.zeros_like(gt)

            train_indices, test_indices = train_test_split(X, train_size=(labeled_ratio), random_state=seed)

            train_indices = [list(t) for t in zip(*train_indices)]
            test_indices = [list(t) for t in zip(*test_indices)]

            train_gt[tuple(train_indices)] = gt[tuple(train_indices)]
            test_gt[tuple(test_indices)] = gt[tuple(test_indices)]

            for idx, xy in enumerate(spatial_loc):
                # print(xy[0],xy[1])
                # print(train_gt[xy[0], xy[1]])
                if train_gt[xy[0], xy[1]] != 0:
                    labeled_idx.append(idx)
                elif test_gt[xy[0], xy[1]] != 0:
                    unlabeled_idx.append(idx)


    elif split_mode == 'disjoint':
        labeled_mask = np.copy(gt)
        unlabeled_mask = np.copy(gt)
        for label in unique_labels:
            mask = gt == label
            for x in range(gt.shape[0]):
                first_half_count = np.count_nonzero(mask[:x, :])
                second_half_count = np.count_nonzero(mask[x:, :])
                try:
                    ratio = first_half_count / second_half_count
                    if 0.9 * labeled_ratio < ratio < 1.1 * labeled_ratio:
                        break
                except ZeroDivisionError:
                    continue

            mask[:x, :] = 0
            labeled_mask[mask] = 0

        unlabeled_mask[labeled_mask > 0] = 0

        for idx, xy in enumerate(spatial_loc):
            if labeled_mask[xy[0], xy[1]] != 0:
                labeled_idx.append(idx)
            elif unlabeled_mask[xy[0], xy[1]] != 0:
                unlabeled_idx.append(idx)

    elif split_mode == 'clustered':
        for label in unique_labels:
            label_indices = np.where(labels == label)[0]
            label_gt = gt[:, label_indices].T

            kmeans = KMeans(n_clusters=1, random_state=seed)
            kmeans.fit(label_gt)
            centroid = kmeans.cluster_centers_[0]

            distances = np.linalg.norm(label_gt - centroid, axis=1)

            sorted_indices = label_indices[np.argsort(distances)]
            num_labeled = int(len(label_indices) * labeled_ratio)
            labeled_idx.extend(sorted_indices[:num_labeled])
            unlabeled_idx.extend(sorted_indices[num_labeled:])

    else:
        raise ValueError(f"{split_mode} sampling is not implemented yet.")

    return labeled_idx, unlabeled_idx


def load_DLPFC_data(id, path='./', dim_RNA=3000, margin=25, labeled_ratio=0.3, split_mode='disjoint', seed=0,):
    # 读取 h5ad 文件
    adata = sc.read_h5ad(os.path.join(path, id, 'sampledata.h5ad'))
    adata.var_names_make_unique()

    # 选择高变异基因，归一化数据，并进行对数转换
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=dim_RNA, check_values=False)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # 读取注释文件
    Ann_df = pd.read_csv('%s/%s/annotation.txt' % (path, id), sep='\t', header=None, index_col=0)
    Ann_df.columns = ['Ground Truth']
    adata.obs['Ground Truth'] = Ann_df.loc[adata.obs_names, 'Ground Truth']

    # 过滤掉没有注释的细胞，并创建副本
    adata = adata[adata.obs.notna().all(axis=1)].copy()
    sc.tl.rank_genes_groups(adata, "Ground Truth", method="wilcoxon")

    # 仅保留高变异基因
    adata = adata[:, adata.var['highly_variable']]

    # 获取 spatial_loc 和 labels
    spatial_loc = adata.obsm['spatial']
    labels = np.array(adata.obs['Ground Truth'].values)

    labels_set = set(labels)
    labels_dict = {k: v for k, v in zip(labels_set, list(range(len(labels_set))))}
    print(labels_dict)
    gt = np.array([labels_dict[i] for i in labels])

    # 使用 split_labeled_data 函数获取有标签和无标签的索引
    labeled_idx, unlabeled_idx = split_labeled_data(spatial_loc, gt, labeled_ratio=labeled_ratio, seed=seed,
                                                    split_mode=split_mode)
    print(f"Number of labeled samples: {len(labeled_idx)}")
    print(f"Number of unlabeled samples: {len(unlabeled_idx)}")
    print(f"Total number of samples: {len(gt)}")
    print(f"Shape of adata.obs: {adata.obs.shape}")

    # print(len(labeled_idx), len(unlabeled_idx), len(gt))
    assert len(labeled_idx) + len(unlabeled_idx) == len(
        gt), "Total number of indices do not match the total number of samples"
    # assert abs(len(unlabeled_idx) / len(gt) - unlabeled_ratio) < 0.1, "Unlabeled ratio is out of acceptable range"

    # 标记有标签样本
    adata = adata.copy()
    adata.obs['is_labeled'] = True
    adata.obs.iloc[unlabeled_idx, adata.obs.columns.get_loc('is_labeled')] = False

    # 保存分割的样本
    base_path = os.path.join('../../SC/DLPFC/', 'Samples', 'split_model_{}_ldr_{}'.format(split_mode, labeled_ratio),
                             'fov_' + str(id))
    Path(base_path).mkdir(parents=True, exist_ok=True)
    all_samples_path = os.path.join(base_path, 'selected_samples_all.csv')

    obs_data = adata.obs[['in_tissue', 'Ground Truth', 'is_labeled']]
    spatial_data = pd.DataFrame(adata.obsm['spatial'], columns=['X', 'Y'], index=adata.obs.index)
    combined_data = pd.concat([obs_data, spatial_data], axis=1)
    combined_data.to_csv(all_samples_path)

    # 读取图像
    if os.path.exists(os.path.join(path, id, 'spatial/full_image.tif')):
        image = cv2.imread(os.path.join(path, id, 'spatial/full_image.tif'))
    elif os.path.exists(os.path.join(path, id, 'spatial/tissue_hires_image.png')):
        image = cv2.imread(os.path.join(path, id, 'spatial/tissue_hires_image.png'))

    # 提取图像 patch
    def extract_patches(spatial, image, margin):
        patches = []
        for py, px in spatial:
            img = image[int(round(px, 0)) - margin:int(round(px, 0)) + margin,
                  int(round(py, 0)) - margin:int(round(py, 0)) + margin]
            if img.shape[0] < 2 * margin or img.shape[1] < 2 * margin:
                pad_height = max(2 * margin - img.shape[0], 0)
                pad_width = max(2 * margin - img.shape[1], 0)
                img = np.pad(img, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant')
            patches.append(img)
        return patches

    RNA_emb = adata.X.toarray()

    # 根据标记提取有标签和无标签的数据
    labeled_idx = adata.obs['is_labeled'] == True
    unlabeled_idx = adata.obs['is_labeled'] == False

    # lb = {
    #     'patchs': extract_patches(spatial_loc[labeled_idx], image, margin),
    #     'RNA_emb': RNA_emb[labeled_idx],
    #     'spatial_loc': spatial_loc[labeled_idx],
    #     'gt': gt[labeled_idx],
    #     'adata': adata[labeled_idx].copy()
    # }
    #
    # ulb = {
    #     'patchs': extract_patches(spatial_loc[unlabeled_idx], image, margin),
    #     'RNA_emb': RNA_emb[unlabeled_idx],
    #     'spatial_loc': spatial_loc[unlabeled_idx],
    #     'gt': gt[unlabeled_idx],
    #     'adata': adata[unlabeled_idx].copy()
    # }
    # all = {
    #     'patchs': extract_patches(spatial_loc, image, margin),
    #     'RNA_emb': RNA_emb,
    #     'spatial_loc': spatial_loc,
    #     'gt': gt,
    #     'adata': adata.copy()
    # }
    #
    # return lb, ulb, all
    return  RNA_emb, spatial_loc, gt, adata, labeled_idx, unlabeled_idx

# def load_DLPFC_data(id, path='./', dim_RNA=3000, margin=25):
#     adata = sc.read_h5ad(os.path.join(path, id, 'sampledata.h5ad'))
#     adata.var_names_make_unique()
#     sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=dim_RNA, check_values=False)
#     sc.pp.normalize_total(adata, target_sum=1e4)
#     sc.pp.log1p(adata)
#     Ann_df = pd.read_csv('%s/%s/annotation.txt' % (path, id), sep='\t', header=None, index_col=0)
#     Ann_df.columns = ['Ground Truth']
#     adata.obs['Ground Truth'] = Ann_df.loc[adata.obs_names, 'Ground Truth']
#     adata = adata[adata.obs.notna().all(axis=1)].copy()
#     sc.tl.rank_genes_groups(adata, "Ground Truth", method="wilcoxon")
#
#     adata = adata[:, adata.var['highly_variable']]
#
#     if os.path.exists(os.path.join(path, id, 'spatial/full_image.tif')):
#         image = cv2.imread(os.path.join(path, id, 'spatial/full_image.tif'))
#     elif os.path.exists(os.path.join(path, id, 'spatial/tissue_hires_image.png')):
#         image = cv2.imread(os.path.join(path, id, 'spatial/tissue_hires_image.png'))
#
#     try:
#         patchs = [image[int(round(px, 0)) - margin:int(round(px, 0)) + margin,
#                   int(round(py, 0)) - margin:int(round(py, 0)) + margin] for py, px in adata.obsm['spatial']]
#
#     except Exception as e:
#
#         patchs = []
#         for py, px in adata.obsm['spatial']:
#             img = image[int(round(px, 0)) - margin:int(round(px, 0)) + margin,
#                   int(round(py, 0)) - margin:int(round(py, 0)) + margin]
#             if img.shape[0] < 2 * margin or img.shape[1] < 2 * margin:
#                 pad_height = max(2 * margin - img.shape[0], 0)
#                 pad_width = max(2 * margin - img.shape[1], 0)
#                 img = np.pad(img, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant')
#             patchs.append(img)
#
#     spatial_loc = adata.obsm['spatial']
#     RNA_emb = adata.X.toarray()
#
#     labels = list(adata.obs['Ground Truth'].values)
#     labels_set = set(labels)
#     labels_dict = {k: v for k, v in zip(labels_set, list(range(len(labels_set))))}
#     gt = np.array([labels_dict[i] for i in labels])
#
#     return patchs, RNA_emb, spatial_loc, gt, adata
