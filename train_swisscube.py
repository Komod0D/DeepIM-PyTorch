import cupy
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import os
import json
from scipy.spatial.transform import Rotation as R

import argparse
import pprint
import time, os, sys
import os.path as osp
import numpy as np
import cv2
import scipy.io
from scipy.spatial.transform import Rotation as R
import glob

import tools._init_paths
from fcn.train_test import test_image
from fcn.config import cfg, cfg_from_file, yaml_from_file, get_output_dir
from datasets.factory import get_dataset
import networks
from networks import FlowNetS

CUDA_DEVICE = 0
MINIBATCH_SIZE = 64


def load_network(network_path):

    weights = torch.load(network_path)
    network = FlowNetS(1).cuda(device=CUDA_DEVICE)
    network.load_state_dict(weights['state_dict'])
    network = torch.nn.DataParallel(network, device_ids=[CUDA_DEVICE]).cuda(device=CUDA_DEVICE)
    cudnn.benchmark = True

    return network


def load_poses(path):
    with open(path, 'r') as f:
        poses = json.load(f)

    return poses


def extract_pose(pose_dict):
    rotation = np.array(pose_dict['cam_R_m2c']).reshape((3, 3))
    translation = np.array(pose_dict['cam_t_m2c'])

    quaternion = R.from_matrix(rotation).as_quat()
    pose = np.concatenate((quaternion, translation))  # TODO: CHECK!!!!!!!!!!!!!!

    return pose


def generate_samples(split='testing'):

    if split not in ['training', 'validation', 'testing']:
        print(f'invalid split name {split}')
        exit(-1)

    points = None
    weights_rot = np.array([1, 1, 1, 1])
    extents = None

    images_cuda = torch.FloatTensor(MINIBATCH_SIZE, 6, width, height).cuda()
    flow_cuda = torch.FloatTensor(MINIBATCH_SIZE, 4, width, height).cuda()

    base_path = '/cvlabdata2/cvlab/datasets_protopap/swisscube/'

    images_list_path = os.path.join(base_path, f'{split}.txt')
    with open(images_list_path, 'r') as f:
        images_list = f.readlines()

    for img_path in images_list:
        full_path = os.path.join(base_path, img_path)
        pose_path = os.path.join(*os.path.split(full_path)[:-2], 'scene_gt.json')
        poses = load_poses(pose_path)
        pose = extract_pose(poses[str(int(os.path.basename(full_path)))])  # TODO check again!!!!!!!

        img = cv2.imread(full_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
