#!/usr/bin/env python3

# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

"""Test a DeepIM on images"""

import cupy
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data

import argparse
import pprint
import time, os, sys
import os.path as osp
import numpy as np
import cv2
import scipy.io
from scipy.io import loadmat
from scipy.spatial.transform import Rotation as R
import glob

import tools._init_paths
from fcn.train_test import test_image
from fcn.config import cfg, cfg_from_file, yaml_from_file, get_output_dir
from datasets.factory import get_dataset
import networks


from utils.blob import pad_im

sys.path.append('../../deepim')
from render_swisscube import Renderer
import json


classes = ['__background__', 'swisscube']

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a PoseCNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--pretrained', dest='pretrained',
                        help='initialize with pretrained checkpoint',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--meta', dest='meta_file',
                        help='optional metadata file', default=None, type=str)
    parser.add_argument('--color', dest='color_name',
                        help='color image pattern',
                        default='*.png', type=str)
    parser.add_argument('--imgdir', dest='imgdir',
                        help='path of the directory with the test images',
                        default='data/images/linemod', type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--network', dest='network_name',
                        help='name of the network',
                        default=None, type=str)
    parser.add_argument('--background', dest='background_name',
                        help='name of the background file',
                        default=None, type=str)

    parser = argparse.ArgumentParser(description='Test a PoseCNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--pretrained', dest='pretrained',
                        help='initialize with pretrained checkpoint',
                        default=None, type=str)
    parser.add_argument('--pretrained_encoder', dest='pretrained_encoder',
                        help='initialize with pretrained encoder checkpoint',
                        default=None, type=str)
    parser.add_argument('--codebook', dest='codebook',
                        help='codebook',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--meta', dest='meta_file',
                        help='optional metadata file', default=None, type=str)
    parser.add_argument('--dataset', dest='dataset_name',
                        help='dataset to train on',
                        default='linemod_test', type=str)
    parser.add_argument('--depth', dest='depth_name',
                        help='depth image pattern',
                        default='*depth.png', type=str)
    parser.add_argument('--color', dest='color_name',
                        help='color image pattern',
                        default='*.bmp', type=str)
    parser.add_argument('--imgdir', dest='imgdir',
                        help='path of the directory with the test images',
                        default='data/images/1b/', type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--network', dest='network_name',
                        help='name of the network',
                        default=None, type=str)
    parser.add_argument('--background', dest='background_name',
                        help='name of the background file',
                        default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args



def init_tensors():

    num = dataset.num_classes
    height = cfg.TRAIN.SYN_HEIGHT
    width = cfg.TRAIN.SYN_WIDTH
    input_blob_color = torch.cuda.FloatTensor(num, 6, height, width).detach()
    image_real_blob_color = torch.cuda.FloatTensor(num, 3, height, width).detach()
    image_tgt_blob_color = torch.cuda.FloatTensor(num, 3, height, width).detach()
    image_src_blob_color = torch.cuda.FloatTensor(num, 3, height, width).detach()
    input_blob_depth = torch.cuda.FloatTensor(num, 6, height, width).detach()
    image_real_blob_depth = torch.cuda.FloatTensor(num, 3, height, width).detach()
    image_tgt_blob_depth = torch.cuda.FloatTensor(num, 3, height, width).detach()
    image_src_blob_depth = torch.cuda.FloatTensor(num, 3, height, width).detach()
    affine_matrices = torch.cuda.FloatTensor(num, 2, 3).detach()
    zoom_factor = torch.cuda.FloatTensor(num, 4).detach()
    flow_blob = torch.cuda.FloatTensor(num, 2, height, width).detach()
    pcloud_tgt_cuda = torch.cuda.FloatTensor(height, width, 3).detach()
    pcloud_src_cuda = torch.cuda.FloatTensor(height, width, 3).detach()
    flow_map_cuda = torch.cuda.FloatTensor(height, width, 2).detach()

    test_data = {'input_blob_color': input_blob_color,
                 'image_real_blob_color': image_real_blob_color,
                 'image_tgt_blob_color': image_tgt_blob_color,
                 'image_src_blob_color': image_src_blob_color,
                 'input_blob_depth': input_blob_depth,
                 'image_real_blob_depth': image_real_blob_depth,
                 'image_tgt_blob_depth': image_tgt_blob_depth,
                 'image_src_blob_depth': image_src_blob_depth,
                 'affine_matrices': affine_matrices,
                 'zoom_factor': zoom_factor,
                 'flow_blob': flow_blob,
                 'pcloud_tgt_cuda': pcloud_tgt_cuda,
                 'pcloud_src_cuda': pcloud_src_cuda,
                 'flow_map_cuda': flow_map_cuda}
    return test_data



def load_network():

    if args.pretrained:
        network_data = torch.load(args.pretrained)
        print("=> using pre-trained network '{}'".format(args.pretrained))
    else:
        network_data = None
        print("no pretrained network specified")
        sys.exit()

    network = networks.__dict__[args.network_name](dataset.num_classes, network_data).cuda(device=cfg.device)
    network = torch.nn.DataParallel(network, device_ids=[0]).cuda(device=cfg.device)
    cudnn.benchmark = True
    network.eval()
    return network

def load_images(obj):
    # list images
    images_color = []
    filename = os.path.join(args.imgdir, args.color_name)

    print(f'getting images from {filename}')
    files = glob.glob(filename)
    for i in range(len(files)):
        filename = files[i]
        images_color.append(filename)
    images_color.sort()

    images_depth = []
    filename = os.path.join(args.imgdir, args.depth_name)
    files = glob.glob(filename)
    for i in range(len(files)):
        filename = files[i]
        images_depth.append(filename)
    images_depth.sort()

    resdir = os.path.join(args.imgdir, 'deepim_results_' + cfg.INPUT)
    if not os.path.exists(resdir):
        os.makedirs(resdir)

    if cfg.TEST.VISUALIZE:
        index_images = np.random.permutation(len(images_color))
    else:
        index_images = range(len(images_color))

    return images_color, images_depth, index_images

if __name__ == '__main__':
    
    intrinsic = np.array([[4000, 0, 1024],
                        [0, 4000, 1024],
                        [0, 0, 1]])
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    print('Using config:')
    pprint.pprint(cfg)

    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)

    # device
    cfg.gpu_id = args.gpu_id
    cfg.device = torch.device('cuda:{:d}'.format(cfg.gpu_id))
    cfg.instance_id = 0
    print('GPU device {:d}'.format(args.gpu_id))

    # dataset
    cfg.MODE = 'TEST'
    cfg.TEST.SYNTHESIZE = False
    dataset = get_dataset(args.dataset_name)
    
    # prepare network
    network = load_network()
    
    for obj in classes:
        images_color, images_depth, index_images = load_images(obj)

        # prepare renderer
        print('loading 3D models')
        cfg.renderer = Renderer()

        # initialize tensors for testing
        test_data = init_tensors()

        # for each image
        for i in index_images:

            im = pad_im(cv2.imread(images_color[i], cv2.IMREAD_COLOR), 16)

            # rescale image if necessary
            if cfg.TEST.SCALES_BASE[0] != 1:
                im_scale = cfg.TEST.SCALES_BASE[0]
                im = pad_im(cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR), 16)
                
            # read initial pose estimation
            name = os.path.basename(images_color[i])

            pose = os.path.join('/cvlabdata2/home/protopap/deepim/data/images/1b/posecnn_results', os.path.basename(images_color[i]) + '.mat')
            pose = loadmat(pose)['pose']
            translation, rotation_q = pose[:3], pose[3:]

            poses = np.concatenate((rotation_q, translation))

            # construct pose input to the network
            poses_input = np.zeros((1, 9), dtype=np.float32)
            # class id in DeepIM starts with 0
            poses_input[0, 1] = 0
            poses_input[0:, 2:] = poses

            # run network 
            im_pose_color, pose_result = test_image(network, dataset, im, im, poses_input, test_data)

            # save result
            if not cfg.TEST.VISUALIZE:
                head, tail = os.path.split(images_color[i])
                filename = os.path.join(resdir, tail + '.mat')
                scipy.io.savemat(filename, pose_result, do_compression=True)
                # rendered image
                filename = os.path.join(resdir, tail + '_render.jpg')
                cv2.imwrite(filename, im_pose_color[:, :, (2, 1, 0)])
