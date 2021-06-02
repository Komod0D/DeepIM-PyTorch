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
from scipy.spatial.transform import Rotation as R
import glob

import tools._init_paths
from fcn.train_test import test_image
from fcn.config import cfg, cfg_from_file, yaml_from_file, get_output_dir
from datasets.factory import get_dataset
import networks

# from ycb_renderer import YCBRenderer :)

from utils.blob import pad_im

sys.path.append('../../deepim')
from bop_toolkit_lib.renderer_adapter import RendererAdapter
import json


classes = [f'{obj:06d}' for obj in range(5, 16)]

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
                        default='*.png', type=str)
    parser.add_argument('--imgdir', dest='imgdir',
                        help='path of the directory with the test images',
                        default='data/images/linemod/%06d/rgb/', type=str)
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


def generate_training(classes):
    batch_size = 64
    for c in classes:
        images_color, images_depth, index_images = load_images(c)

        img_batch = []
        info_batch = []
        for path in images_color:
            if len(img_batch) < 64:
                img_batch.append(pad_im(cv2.imread(path, cv2.IMREAD_COLOR), 16)[np.newaxis, :])
            else:
                im_tensor = torch.from_numpy(np.vstack(img_batch)).float() / 255.0
                im_tensor -= cfg.PIXEL_MEAN
                im_cuda_color = im_tensor.cuda()

                im_cuda_depth = im_cuda_color.clone().detach()

                # construct the meta data
                K = dataset._intrinsic_matrix
                Kinv = np.linalg.pinv(K)
                meta_data_blob = np.zeros(18, dtype=np.float32)
                meta_data_blob[0:9] = K.flatten()
                meta_data_blob[9:18] = Kinv.flatten()
                label_blob = np.zeros((dataset.num_classes, height, width), dtype=np.float32)
                gt_boxes = np.zeros((num, 5), dtype=np.float32)
                im_info = np.array([im_color.shape[0], im_color.shape[1], cfg.TRAIN.SCALES_BASE[0]], dtype=np.float32)

                poses = results[str(num)][0]

                rotation = np.array(poses['cam_R_m2c']).reshape((3, 3))
                dr = R.from_euler('xyz', np.random.random(size=(3,)) * 0.1 - 0.05).as_matrix()
                rotation_d = dr @ rotation

                translation = np.array(poses['cam_t_m2c'])
                t_dev = np.abs(translation) / 20
                dt = np.random.random(size=(3,)) * t_dev
                translation_d = translation + dt

                # construct pose input to the network
                poses_input = np.zeros((1, 9), dtype=np.float32)
                # class id in DeepIM starts with 0
                poses_input[0, 1] = int(obj) - 1
                poses_input[0:, 2:] = poses

                yield img_batch
                img_batch.clear()




            sample = {'image_color': im_cuda_color.unsqueeze(0),
                  'image_depth': im_cuda_depth.unsqueeze(0),
                  'meta_data': torch.from_numpy(meta_data_blob[np.newaxis,:]),
                  'label_blob': torch.from_numpy(label_blob[np.newaxis,:]),
                  'poses': torch.from_numpy(poses_input[np.newaxis, :]),
                  'extents': torch.from_numpy(dataset._extents[np.newaxis,:]),
                  'points': torch.from_numpy(dataset._point_blob[np.newaxis,:]),
                  'gt_boxes': torch.from_numpy(gt_boxes[np.newaxis,:]),
                  'poses_result': torch.from_numpy(poses_input[np.newaxis,:]),
                  'im_info': torch.from_numpy(im_info[np.newaxis,:]),
                      'mask': torch.from_numpy(np.ones_like(im_color))}


            im = pad_im(cv2.imread(images_color[i], cv2.IMREAD_COLOR), 16)
            print(images_color[i])
            if len(images_depth) > 0 and osp.exists(images_depth[i]):
                depth = pad_im(cv2.imread(images_depth[i], cv2.IMREAD_UNCHANGED), 16)
                depth = depth.astype('float') / 1000.0
                print(images_depth[i])
            else:
                depth = None
                print('no depth image')

            # rescale image if necessary
            if cfg.TEST.SCALES_BASE[0] != 1:
                im_scale = cfg.TEST.SCALES_BASE[0]
                im = pad_im(cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR), 16)
                if depth is not None:
                    depth = pad_im(cv2.resize(depth, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_NEAREST), 16)

            # read initial pose estimation
            name = os.path.basename(images_color[i])


            num = int(name[:-4])
            poses = results[str(num)][0]

            rotation = np.array(poses['cam_R_m2c']).reshape((3,3))
            dr = R.from_euler('xyz', np.random.random(size=(3,)) * 0.1 - 0.05).as_matrix()
            print(f'disturbing rotation by {dr}')
            rotation = dr @ rotation

            translation = np.array(poses['cam_t_m2c'])
            t_dev = np.abs(translation) / 20
            dt = np.random.random(size=(3,)) * t_dev
            translation += dt
            print(f'disturbing translation by {dt}')


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
    filename = os.path.join(args.imgdir % int(obj), args.color_name)

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
    K = np.array([572.4114, 0.0, 325.2611, 0.0, 573.57043, 242.04899, 0.0, 0.0, 1.0]).reshape((3, 3))
    intrinsic = K
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if len(cfg.TEST.CLASSES) == 0:
        cfg.TEST.CLASSES = cfg.TRAIN.CLASSES

    if args.meta_file is not None:
        meta = yaml_from_file(args.meta_file)
        # overwrite test classes
        print(meta)
        if 'ycb_ids' in meta:
            cfg.TEST.CLASSES = [0]
            for i in meta.ycb_ids:
                cfg.TEST.CLASSES.append(i)
            print('TEST CLASSES:', cfg.TEST.CLASSES)
        if 'INTRINSICS' in meta:
            cfg.INTRINSICS = meta['INTRINSICS']

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

    # overwrite intrinsics
    if len(cfg.INTRINSICS) > 0:
        K = np.array(cfg.INTRINSICS).reshape(3, 3)
        if cfg.TEST.SCALES_BASE[0] != 1:
            scale = cfg.TEST.SCALES_BASE[0]
            K[0, 0] *= scale
            K[0, 2] *= scale
            K[1, 1] *= scale
            K[1, 2] *= scale
        dataset._intrinsic_matrix = K
        print(f"Intrinsic matrix: \n{dataset._intrinsic_matrix}")

    
    # prepare network
    network = load_network()
    
    for obj in classes:
        images_color, images_depth, index_images = load_images(obj)



        # prepare renderer
        print('loading 3D models')
        cfg.renderer = RendererAdapter(width=cfg.TRAIN.SYN_WIDTH, height=cfg.TRAIN.SYN_HEIGHT)
        cfg.renderer.load_object(int(obj))


        # initialize tensors for testing
        test_data = init_tensors()

        
        result_file = f'/cvlabdata2/cvlab/datasets_protopap/linemod/test/{int(obj):06d}/scene_gt.json'
        print(f'fetching poses from {result_file}')        
        with open(result_file, 'r') as f:
            results = json.load(f)
        
        # for each image
        for i in index_images:
            im = pad_im(cv2.imread(images_color[i], cv2.IMREAD_COLOR), 16)
            print(images_color[i])
            if len(images_depth) > 0 and osp.exists(images_depth[i]):
                depth = pad_im(cv2.imread(images_depth[i], cv2.IMREAD_UNCHANGED), 16)
                depth = depth.astype('float') / 1000.0
                print(images_depth[i])
            else:
                depth = None
                print('no depth image')

            # rescale image if necessary
            if cfg.TEST.SCALES_BASE[0] != 1:
                im_scale = cfg.TEST.SCALES_BASE[0]
                im = pad_im(cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR), 16)
                if depth is not None:
                    depth = pad_im(cv2.resize(depth, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_NEAREST), 16)

            # read initial pose estimation
            name = os.path.basename(images_color[i])


            num = int(name[:-4])
            poses = results[str(num)][0]

            rotation = np.array(poses['cam_R_m2c']).reshape((3,3))
            dr = R.from_euler('xyz', np.random.random(size=(3,)) * 0.1 - 0.05).as_matrix()
            print(f'disturbing rotation by {dr}')
            rotation_d = dr @ rotation
            
            translation = np.array(poses['cam_t_m2c'])
            t_dev = np.abs(translation) / 20
            dt = np.random.random(size=(3,)) * t_dev
            translation_d = translation + dt
            print(f'disturbing translation by {dt}')
            

            rotation_q = scipy.spatial.transform.Rotation.from_matrix(rotation_d).as_quat()

            poses = np.concatenate((rotation_q, translation_d))

            # construct pose input to the network
            poses_input = np.zeros((1, 9), dtype=np.float32)
            # class id in DeepIM starts with 0
            poses_input[0, 1] = int(obj) - 1
            poses_input[0:, 2:] = poses
            
            background = np.zeros((3, 480, 640))
            
            import itertools

            train_loader = generate_training()
            background_loader = itertools.repeat(background)
            optimizer = torch.optim.Adam(network.parameters(), lr=0.01)
            epoch = 0
            num_iterations = 4
            
