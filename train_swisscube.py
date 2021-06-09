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

from render_swisscube import Renderer
from fcn.train_test import process_sample, _compute_pose_target
from fcn.multiscaleloss import multiscaleEPE, realEPE

CUDA_DEVICE = 0
MINIBATCH_SIZE = 64


def load_network(network_path):

    weights = torch.load(network_path)
    network = FlowNetS(1).cuda(device=CUDA_DEVICE)
    network.load_state_dict(weights['state_dict'])
    network = torch.nn.DataParallel(network, device_ids=[CUDA_DEVICE]).cuda(device=CUDA_DEVICE)
    cudnn.benchmark = True

    return network



def init_tensors():


    num = MINIBATCH_SIZE
    height, width = 480, 640
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

    train_data = {'input_blob_color': input_blob_color,
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


    return train_data


def load_poses(path):
    with open(path, 'r') as f:
        poses = json.load(f)

    return poses


def extract_pose(pose_dict):
    rotation = np.array(pose_dict['cam_R_m2c']).reshape((3, 3))
    translation = np.array(pose_dict['cam_t_m2c'])

    quaternion = R.from_matrix(rotation).as_quat()
    pose = np.concatenate((quaternion, translation))  # TODO: CHECK!!!!!!!!!!!!!!
    pose_deepim = np.zeros((1, 9))
    pose_deepim[2:] = pose
    return pose


def generate_samples(split='testing'):

    if split not in ['training', 'validation', 'testing']:
        print(f'invalid split name {split}')
        exit(-1)

    points = None
    weights_rot = np.array([1, 1, 1, 1])
    extents = None

    width, height = 640, 480
    images_cuda = torch.FloatTensor(MINIBATCH_SIZE, 6, width, height).cuda()
    flow_cuda = torch.FloatTensor(MINIBATCH_SIZE, 4, width, height).cuda()

    base_path = '/cvlabdata2/cvlab/datasets_protopap/swisscube/'

    images_list_path = os.path.join(base_path, f'{split}.txt')
    with open(images_list_path, 'r') as f:
        images_list = f.readlines()

    images = []
    poses_est = []
    for img_path in images_list:
        full_path = os.path.join(base_path, img_path.strip())
        pose_path = os.path.join(os.path.dirname(os.path.dirname(full_path))), 'scene_gt.json')
        poses = load_poses(pose_path)
        pose = extract_pose(poses[str(int(os.path.basename(full_path)))])  # TODO check again!!!!!!!

        img = cv2.imread(full_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 640), cv2.INTER_AREA)
        img = img[80:560]
        img = np.transpose(img, (2, 1, 0))[np.newaxis, :]
        images.append(img)


        if len(images) == MINIBATCH_SIZE:
            images_cpu = np.vstack(images)
            images_cuda = torch.from_numpy(images_cpu).cuda()
            

            poses_est.append(pose)
            sample = {'image_color': images_cuda,
                  'image_depth': images_cuda,
                  'meta_data': torch.from_numpy(meta_data_blob[np.newaxis,:]),
                  'label_blob': torch.from_numpy(label_blob[np.newaxis,:]),
                  'poses': torch.from_numpy(poses_est),
                  'extents': torch.from_numpy(dataset._extents[np.newaxis,:]),
                  'points': torch.from_numpy(dataset._point_blob[np.newaxis,:]),
                  'poses_result': torch.from_numpy(poses_est[np.newaxis,:])}
            yield sample

def train(gen_samples, network, optimizer):
    
    train_data = init_tensors()
    for sample in gen_samples:
       
        end = time.time()
        inputs, inputs_depth, flow, poses_src, poses_tgt, \
            weights_rot, extents, points, affine_matrices, zoom_factor, vdata = \
            process_sample(sample, poses_est, train_data)
        data_time.update(time.time() - end)

        # measure data loading time
        poses_src = poses_src.cuda().detach()
        poses_tgt = poses_tgt.cuda().detach()
        weights_rot = weights_rot.cuda().detach()
        extents = extents.cuda().detach()
        points = points.cuda().detach()

        # zoom in image
        grids = nn.functional.affine_grid(affine_matrices, inputs.size())
        input_zoom = nn.functional.grid_sample(inputs, grids).detach()

        # zoom in flow
        flow_zoom = nn.functional.grid_sample(flow, grids)
        for k in range(flow_zoom.shape[0]):
            flow_zoom[k, 0, :, :] /= affine_matrices[k, 0, 0] * 20.0
            flow_zoom[k, 1, :, :] /= affine_matrices[k, 1, 1] * 20.0
        
        output, loss_pose_tensor, quaternion_delta_var, translation_var = \
            network(input_zoom, weights_rot, poses_src, poses_tgt, extents, points, zoom_factor)

        
        vdata_pose = vdata['pose_src']
        quaternion_delta = quaternion_delta_var.cpu().detach().numpy()
        translation = translation_var.cpu().detach().numpy()
        poses_est, error_rot, error_trans = \
                _compute_pose_target(quaternion_delta, translation, vdata_pose, vdata['pose_tgt'])

        # losses
        loss_pose = torch.mean(loss_pose_tensor)
        loss_flow = 0.1 * multiscaleEPE(output, flow_zoom)
        flow2_EPE = realEPE(output[0], flow_zoom)
        loss = loss_pose + loss_flow

        # compute gradient and do optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time = time.time() - end

        print('epoch: [%d/%d][%d/%d], iter %d, loss %.4f, l_pose %.4f (%.2f, %.2f), l_flow %.4f, lr %.6f, data time %.2f, batch time %.2f' \
            % (epoch, cfg.epochs, i, epoch_size, j+1, loss, loss_pose, error_rot, error_trans, loss_flow, \
              optimizer.param_groups[0]['lr'], data_time.val, batch_time.val))

                
                
if __name__ == '__main__':
    
    cfg_from_file('experiments/cfgs/swisscube.yml')

    cfg.renderer = Renderer(synthetic=True)
    network_path = 'data/checkpoints/from_deepim.pth'
    network = load_network(network_path)
    
    
    param_groups = network.params() # TODO fix
    optimizer = torch.optim.SGD(param_groups, cfg.TRAIN.LEARNING_RATE, momentum=cfg.TRAIN.MOMENTUM)
    


    epochs = 25
    cfg.epochs = epochs
    for i in range(epochs):
        train(generate_samples('training'), network, optimizer)
