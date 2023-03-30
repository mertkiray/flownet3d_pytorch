#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import random
import sys
import glob
import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import minkowski
import open3d as o3d
import augmentation as t


# Part of the code is referred from: https://github.com/charlesq34/pointnet

def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))


def load_data(partition):
    # download()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'dataset/')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.05):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    return pointcloud


def farthest_subsample_points(pointcloud1, pointcloud2, num_subsampled_points=768):
    pointcloud1 = pointcloud1.T
    pointcloud2 = pointcloud2.T
    num_points = pointcloud1.shape[0]
    nbrs1 = NearestNeighbors(n_neighbors=num_subsampled_points, algorithm='auto',
                             metric=lambda x, y: minkowski(x, y), n_jobs=1).fit(pointcloud1)
    random_p1 = np.random.random(size=(1, 3)) + np.array([[500, 500, 500]]) * np.random.choice([1, -1, 1, -1])
    idx1 = nbrs1.kneighbors(random_p1, return_distance=False).reshape((num_subsampled_points,))
    nbrs2 = NearestNeighbors(n_neighbors=num_subsampled_points, algorithm='auto',
                             metric=lambda x, y: minkowski(x, y), n_jobs=1).fit(pointcloud2)
    random_p2 = random_p1 #np.random.random(size=(1, 3)) + np.array([[500, 500, 500]]) * np.random.choice([1, -1, 2, -2])
    idx2 = nbrs2.kneighbors(random_p2, return_distance=False).reshape((num_subsampled_points,))
    return pointcloud1[idx1, :].T, pointcloud2[idx2, :].T


class ModelNet40(Dataset):
    def __init__(self, num_points, num_subsampled_points = 768, partition='train', gaussian_noise=False, unseen=False, factor=4):
        self.data, self.label = load_data(partition)
        self.num_points = num_points
        self.partition = partition
        self.gaussian_noise = gaussian_noise
        self.unseen = unseen
        self.label = self.label.squeeze()
        self.factor = factor
        self.num_subsampled_points = num_subsampled_points
        if num_points != num_subsampled_points:
            self.subsampled = True
        else:
            self.subsampled = False
        if self.unseen:
            ######## simulate testing on first 20 categories while training on last 20 categories
            if self.partition == 'test':
                self.data = self.data[self.label>=20]
                self.label = self.label[self.label>=20]
            elif self.partition == 'train':
                self.data = self.data[self.label<20]
                self.label = self.label[self.label<20]

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        # if self.gaussian_noise:
        #     pointcloud = jitter_pointcloud(pointcloud)
        if self.partition != 'train':
            np.random.seed(item)
        anglex = np.random.uniform() * np.pi / self.factor
        angley = np.random.uniform() * np.pi / self.factor
        anglez = np.random.uniform() * np.pi / self.factor

        cosx = np.cos(anglex)
        cosy = np.cos(angley)
        cosz = np.cos(anglez)
        sinx = np.sin(anglex)
        siny = np.sin(angley)
        sinz = np.sin(anglez)
        Rx = np.array([[1, 0, 0],
                        [0, cosx, -sinx],
                        [0, sinx, cosx]])
        Ry = np.array([[cosy, 0, siny],
                        [0, 1, 0],
                        [-siny, 0, cosy]])
        Rz = np.array([[cosz, -sinz, 0],
                        [sinz, cosz, 0],
                        [0, 0, 1]])
        # 生成旋转矩阵
        R_ab = Rx.dot(Ry).dot(Rz)
        R_ba = R_ab.T
        # 生成平移向量
        translation_ab = np.array([np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5),
                                   np.random.uniform(-0.5, 0.5)])
        translation_ba = -R_ba.dot(translation_ab)

        pointcloud1 = pointcloud.T
        rotation_ab = Rotation.from_euler('zyx', [anglez, angley, anglex])
        pointcloud2 = rotation_ab.apply(pointcloud1.T).T + np.expand_dims(translation_ab, axis=1)

        euler_ab = np.asarray([anglez, angley, anglex])
        euler_ba = -euler_ab[::-1]

        pointcloud1 = np.random.permutation(pointcloud1.T).T
        pointcloud2 = np.random.permutation(pointcloud2.T).T

        if self.gaussian_noise:
            pointcloud1 = jitter_pointcloud(pointcloud1)
            pointcloud2 = jitter_pointcloud(pointcloud2)

        if self.subsampled:
            pointcloud1, pointcloud2 = farthest_subsample_points(pointcloud1, pointcloud2,
                                                                 num_subsampled_points=self.num_subsampled_points)

        return pointcloud1.astype('float32'), pointcloud2.astype('float32'), R_ab.astype('float32'), \
               translation_ab.astype('float32'), R_ba.astype('float32'), translation_ba.astype('float32'), \
               euler_ab.astype('float32'), euler_ba.astype('float32')

    def __len__(self):
        return self.data.shape[0]


class SceneflowDataset(Dataset):
    ELASTIC_DISTORT_PARAMS = ((0.2, 0.4), (0.8, 1.6))
    ROTATION_AXIS = 'z'

    def __init__(self, args, partition='train'):
        self.npoints = args.num_points
        self.partition = partition
        self.root = args.dataset_path
        if self.partition=='train':
            self.datapath = glob.glob(os.path.join(self.root, 'TRAIN*.npz'))
        else:
            self.datapath = glob.glob(os.path.join(self.root, 'TEST*.npz'))
        self.cache = {}
        self.cache_size = 30000
        self.flow_aug = args.flow_aug
        self.use_color = args.use_color
        self.flow_aug_type = args.flow_aug_type
        self.use_aug = args.use_aug

        if self.use_aug:
            input_transform = t.Compose([
                t.ElasticDistortion(self.ELASTIC_DISTORT_PARAMS),
                t.RandomHorizontalFlip(self.ROTATION_AXIS, is_temporal=False),
            ])
            self.aug = t.Compose([input_transform])

        ###### deal with one bad datapoint with nan value
        self.datapath = [d for d in self.datapath if 'TRAIN_C_0140_left_0006-0' not in d]
        ######
        print(self.partition, ': ',len(self.datapath))

    def __getitem__(self, index):
        if index in self.cache:
            pos1, pos2, color1, color2, flow, mask1 = self.cache[index]
        else:
            fn = self.datapath[index]
            with open(fn, 'rb') as fp:
                data = np.load(fp)
                pos1 = data['points1'].astype('float32')
                pos2 = data['points2'].astype('float32')
                color1 = data['color1'].astype('float32')
                color2 = data['color2'].astype('float32')
                flow = data['flow'].astype('float32')
                mask1 = data['valid_mask1']

            if len(self.cache) < self.cache_size:
                self.cache[index] = (pos1, pos2, color1, color2, flow, mask1)

        if self.partition == 'train':
            n1 = pos1.shape[0]
            sample_idx1 = np.random.choice(n1, self.npoints, replace=False)
            n2 = pos2.shape[0]
            sample_idx2 = np.random.choice(n2, self.npoints, replace=False)

            pos1 = pos1[sample_idx1, :]
            pos2 = pos2[sample_idx2, :]
            color1 = color1[sample_idx1, :]
            color2 = color2[sample_idx2, :]
            flow = flow[sample_idx1, :]
            mask1 = mask1[sample_idx1]
        else:
            pos1 = pos1[:self.npoints, :]
            pos2 = pos2[:self.npoints, :]
            color1 = color1[:self.npoints, :]
            color2 = color2[:self.npoints, :]
            flow = flow[:self.npoints, :]
            mask1 = mask1[:self.npoints]

        if not self.use_color:
            color1 = np.zeros_like(color1)
            color2 = np.zeros_like(color2)

        if self.use_aug:
            pos1, flow = self.aug(pos1, flow)

        pos1_center = np.mean(pos1, 0)
        pos1 -= pos1_center
        pos2 -= pos1_center

        return pos1, pos2, color1, color2, flow, mask1
        # if self.partition == 'train' and self.flow_aug:
        #     if random.random() <= 0.5:
        #         return pos1, pos2, color1, color2, flow, mask1
        #     else:
        #         # The augmentation
        #         # select 2 random points
        #         idx_knn = np.random.choice(self.npoints, 5, replace=False)
        #         # create open3d kdtree
        #         pcd = o3d.geometry.PointCloud()
        #         pcd.points = o3d.utility.Vector3dVector(pos1)
        #         kdtree = o3d.geometry.KDTreeFlann(pcd)
        #
        #         if self.flow_aug_type == 'random':
        #             # find the nearest neighbor of the 2 random points
        #             # max flow min flow
        #             max_flow_x = np.max(flow[:, 0])
        #             min_flow_x = np.min(flow[:, 0])
        #             max_flow_y = np.max(flow[:, 1])
        #             min_flow_y = np.min(flow[:, 1])
        #             max_flow_z = np.max(flow[:, 2])
        #             min_flow_z = np.min(flow[:, 2])
        #             for i in range(5):
        #                 [_, idx, _] = kdtree.search_radius_vector_3d(pos1[idx_knn[i], :], 0.5)
        #                 idx = np.array(idx)
        #                 # create random x, y, z flows between min and max
        #                 flow_x = np.random.uniform(min_flow_x, max_flow_x, idx.shape[0])
        #                 flow_y = np.random.uniform(min_flow_y, max_flow_y, idx.shape[0])
        #                 flow_z = np.random.uniform(min_flow_z, max_flow_z, idx.shape[0])
        #                 flow_idx_augmented = np.zeros((idx.shape[0], 3))
        #                 flow_idx_augmented[:, 0] = flow_x
        #                 flow_idx_augmented[:, 1] = flow_y
        #                 flow_idx_augmented[:, 2] = flow_z
        #                 flow[idx[0:], :] = flow_idx_augmented
        #             pos2 = pos1 + flow
        #             color2 = color1
        #             mask1 = np.ones_like(mask1)
        #             return pos1, pos2, color1, color2, flow, mask1
        #
        #         elif self.flow_aug_type == 'replace':
        #             for i in range(5):
        #                 [_, idx, _] = kdtree.search_radius_vector_3d(pos1[idx_knn[i], :], 0.5)
        #                 idx = np.array(idx)
        #                 if idx.shape[0] == 0:
        #                     continue
        #                 random_index = random.randint(0, self.__len__() - 1)
        #                 if random_index in self.cache:
        #                     pos1_random, pos2_random, color1_random, color2_random, flow_random, mask1_random = self.cache[random_index]
        #                 else:
        #                     fn_random = self.datapath[random_index]
        #                     with open(fn_random, 'rb') as fp_random:
        #                         data_random = np.load(fp_random)
        #                         pos1_random = data_random['points1'].astype('float32')
        #                         pos2_random = data_random['points2'].astype('float32')
        #                         color1_random = data_random['color1'].astype('float32')
        #                         color2_random = data_random['color2'].astype('float32')
        #                         flow_random = data_random['flow'].astype('float32')
        #                         mask1_random = data_random['valid_mask1']
        #
        #                 if not self.use_color:
        #                     color1_random = np.zeros_like(color1_random)
        #                     color2_random = np.zeros_like(color2_random)
        #
        #                 pcd_random = o3d.geometry.PointCloud()
        #                 pcd_random.points = o3d.utility.Vector3dVector(pos1_random)
        #                 kdtree_random = o3d.geometry.KDTreeFlann(pcd_random)
        #                 random_point = np.random.choice(pos1_random.shape[0], 1, replace=False)
        #                 [_, idx_random, _] = kdtree_random.search_knn_vector_3d(pos1_random[random_point[0], :], idx.shape[0])
        #                 idx_random = np.array(idx_random)
        #                 pos1[idx[0:], :] = pos1_random[idx_random[0:], :]
        #                 pos2[idx[0:], :] = pos2_random[idx_random[0:], :]
        #                 color1[idx[0:], :] = color1_random[idx_random[0:], :]
        #                 color2[idx[0:], :] = color2_random[idx_random[0:], :]
        #                 flow[idx[0:], :] = flow_random[idx_random[0:], :]
        #                 mask1[idx[0:]] = mask1_random[idx_random[0:]]
        #             return pos1, pos2, color1, color2, flow, mask1
        # else:
        #     return pos1, pos2, color1, color2, flow, mask1

    def __len__(self):
        return len(self.datapath)


if __name__ == '__main__':
    train = ModelNet40(1024)
    test = ModelNet40(1024, 'test')
    for data in train:
        print(data[0].shape)
        break
