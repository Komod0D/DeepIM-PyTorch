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


def load_network():

    network_path = 'data/checkpoints/ycb_object/flownets_ycb_object_20objects_color_self_supervision_epoch_10.checkpoint.pth'

    weights = torch.load(network_path)
    network = FlowNetS(1, weights).cuda(device=CUDA_DEVICE)
    network = torch.nn.DataParallel(network, device_ids=[CUDA_DEVICE]).cuda(device=CUDA_DEVICE)
    cudnn.benchmark = True

    return network


def load_poses(path):
    with open(path, 'r') as f:
        poses = json.load(f)

    return poses


def extract_pose(rotation_list, translation_list):
    rotation = np.array(rotation_list).reshape((3, 3))
    translation = np.array(translation_list)

    quaternion = R.from_matrix(rotation).as_quat()

    pose = np.concatenate((quaternion, translation))  # TODO: CHECK!!!!!!!!!!!!!!

    return pose

def generate_samples():

    path = '/cvlabdata2/cvlab/datasets_protopap/swisscube'
    folders = os.listdir(path)
    for folder in folders:
        sub_folder = folder[folder.index('_') + 1:]
        full_path = os.path.join(path, folder, sub_folder)
        poses = load_poses(full_path)
        images_path = os.path.join(full_path, 'rgb')
        for image_name in os.listdir(images_path):
            poses = poses[]


        yield sample