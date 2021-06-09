import json
import os
import sys
import time

import cupy
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data

import tools._init_paths
from fcn.config import cfg, cfg_from_file
from fcn.multiscaleloss import multiscaleEPE
from fcn.train_test import _compute_pose_target
from networks.FlowNetS import FlowNetS
from scipy.spatial.transform import Rotation as R
from utils.se3 import se3_mul, se3_inverse

from render_swisscube import Renderer

CUDA_DEVICE = 0
MINIBATCH_SIZE = 32

width, height = 640, 480
r = Renderer(synthetic=True)
intrinsic = r.intrinsic

points = np.genfromtxt('data/models/swisscube/points_new.xyz').astype(np.float32)
points = points @ R.from_euler('x', 90, degrees=True).as_matrix().T


weights_rot = np.array([[1, 1, 1, 1]], dtype=np.float32)
extents = np.zeros((1, 3), dtype=np.float32)
extents[0] = 2 * np.max(np.absolute(points), axis=0)

tweights_rot = torch.from_numpy(weights_rot).cuda().detach()
textents = torch.from_numpy(extents[np.newaxis, :]).cuda().detach()
tpoints = torch.from_numpy(points[np.newaxis, np.newaxis, :]).cuda().detach()
print(textents.shape)
print(tpoints.shape)

threadsperblock = (32, 32, 1)
blockspergrid_x = np.ceil(height / threadsperblock[0])
blockspergrid_y = np.ceil(width / threadsperblock[1])
blockspergrid = (int(blockspergrid_x), int(blockspergrid_y), 1)

class Stream:
    ptr = torch.cuda.current_stream().cuda_stream

if sys.version_info[0] < 3:
    @cupy.util.memoize(for_each_device=True)
    def cunnex(strFunction):
        return cupy.cuda.compile_with_cache(globals()[strFunction]).get_function(strFunction)
else:
    @cupy._util.memoize(for_each_device=True)
    def cunnex(strFunction):
        return cupy.cuda.compile_with_cache(globals()[strFunction]).get_function(strFunction)

compute_flow = '''
extern "C" __global__ void compute_flow(float* pc_tgt, float* pc_src, float* flow_map, float* RT, 
                             float fx, float fy, float px, float py, int width, int height)
{
  const int y = threadIdx.x + blockDim.x * blockIdx.x;
  const int x = threadIdx.y + blockDim.y * blockIdx.y;

  if (x < width && y < height) 
  {
    flow_map[(y * width + x) * 2] = 0;
    flow_map[(y * width + x) * 2 + 1] = 0;
    float X = pc_src[(y * width + x) * 3];
    float Y = pc_src[(y * width + x) * 3 + 1];
    float Z = pc_src[(y * width + x) * 3 + 2];
    if (Z > 0)
    {
      float vx = RT[0] * X + RT[1] * Y + RT[2] * Z + RT[3];
      float vy = RT[4] * X + RT[5] * Y + RT[6] * Z + RT[7];
      float vz = RT[8] * X + RT[9] * Y + RT[10] * Z + RT[11];

      // projection
      float w_proj = fx * (vx / vz) + px;
      float h_proj = fy * (vy / vz) + py;
      float z_proj = vz;
      int w_proj_i = roundf(w_proj);
      int h_proj_i = roundf(h_proj);

      if (w_proj_i >= 0 && w_proj_i < width && h_proj_i >= 0 && h_proj_i < height)
      {
        float z_tgt = pc_tgt[(h_proj_i * width + w_proj_i) * 3 + 2];
        if (fabs(z_proj - z_tgt) < 3E-3) 
        {
          flow_map[(y * width + x) * 2] = w_proj - x;
          flow_map[(y * width + x) * 2 + 1] = h_proj - y;
        }
      }
    }  
  }
}
'''

avg_t = np.array([-5.52700769e+00, 5.10048243e+01, 3.25467835e+02])


def fun_compute_flow(pcloud_tgt, pcloud_src, flow_cuda, pose):
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    px = intrinsic[0, 2]
    py = intrinsic[1, 2]
    flow_map_cuda = torch.cuda.FloatTensor(height, width, 2).detach()
    with torch.cuda.device_of(flow_cuda):
        cunnex('compute_flow')(
            grid=blockspergrid,
            block=threadsperblock,
            args=[pcloud_tgt.data_ptr(),
                  pcloud_src.data_ptr(),
                  flow_map_cuda.data_ptr(),
                  pose.data_ptr(),
                  fx, fy, px, py, width, height],
            stream=Stream)
    flow_cuda.copy_(flow_map_cuda.permute(2, 0, 1))


def process(poses_src, poses_tgt, seg_src, seg_tgt, flows):

    num = poses_src.shape[0]
    affine_matrices = torch.cuda.FloatTensor(num, 2, 3).detach()
    zoom_factor = torch.cuda.FloatTensor(num, 4).detach()

    x3d = np.ones((4, points.shape[0]), dtype=np.float32)
    x3d[0, :] = points[:, 0]
    x3d[1, :] = points[:, 1]
    x3d[2, :] = points[:, 2]

    for j in range(num):

        cfg.renderer.set_pose(poses_src[j])
        cfg.renderer.render(seg_src[j])

        cfg.renderer.set_pose(poses_tgt[j])
        cfg.renderer.render(seg_tgt[j])

        RT_tgt = np.zeros((3, 4), dtype=np.float32)
        RT_src = np.zeros((3, 4), dtype=np.float32)

        RT_tgt[:3, :3] = R.from_quat(poses_tgt[j, 2:6]).as_matrix()
        RT_tgt[:, 3] = poses_tgt[j, 6:]

        # compute box
        RT_src[:3, :3] = R.from_quat(poses_src[j, 2:6]).as_matrix()
        RT_src[:, 3] = poses_src[j, 6:]
        x2d = np.matmul(intrinsic, np.matmul(RT_src, x3d))
        x2d[0, :] = np.divide(x2d[0, :], x2d[2, :])
        x2d[1, :] = np.divide(x2d[1, :], x2d[2, :])
        obj_imgn_start_x = np.min(x2d[0, :])
        obj_imgn_start_y = np.min(x2d[1, :])
        obj_imgn_end_x = np.max(x2d[0, :])
        obj_imgn_end_y = np.max(x2d[1, :])
        obj_imgn_c = np.dot(intrinsic, poses_src[j, 6:])
        zoom_c_x = obj_imgn_c[0] / obj_imgn_c[2]
        zoom_c_y = obj_imgn_c[1] / obj_imgn_c[2]

        # mask region
        ratio = float(height) / float(width)
        left_dist = zoom_c_x - obj_imgn_start_x
        right_dist = obj_imgn_end_x - zoom_c_x
        up_dist = zoom_c_y - obj_imgn_start_y
        down_dist = obj_imgn_end_y - zoom_c_y
        crop_height = np.max([ratio * right_dist, ratio * left_dist, up_dist, down_dist]) * 2 * 1.4
        crop_width = crop_height / ratio

        # affine transformation for PyTorch
        x1 = (zoom_c_x - crop_width / 2) * 2 / width - 1
        x2 = (zoom_c_x + crop_width / 2) * 2 / width - 1
        y1 = (zoom_c_y - crop_height / 2) * 2 / height - 1
        y2 = (zoom_c_y + crop_height / 2) * 2 / height - 1

        pts1 = np.float32([[x1, y1], [x1, y2], [x2, y1]])
        pts2 = np.float32([[-1, -1], [-1, 1], [1, -1]])
        affine_matrix = torch.tensor(cv2.getAffineTransform(pts2, pts1))
        affine_matrices[j].copy_(affine_matrix)
        zoom_factor[j, 0] = affine_matrix[0, 0]
        zoom_factor[j, 1] = affine_matrix[1, 1]
        zoom_factor[j, 2] = affine_matrix[0, 2]
        zoom_factor[j, 3] = affine_matrix[1, 2]

        pose = torch.tensor(se3_mul(RT_tgt, se3_inverse(RT_src))).cuda().float()
        fun_compute_flow(seg_tgt[j], seg_src[j], flows[j], pose)

    return affine_matrices, zoom_factor


def load_network(network_path):

    weights = torch.load(network_path)
    network = FlowNetS(1, batchNorm=False).cuda(device=CUDA_DEVICE)
    network.load_state_dict(weights['state_dict'])
    network = torch.nn.DataParallel(network, device_ids=[CUDA_DEVICE]).cuda(device=CUDA_DEVICE)
    cudnn.benchmark = True
    network.train()

    return network


def load_poses(path):
    with open(path, 'r') as f:
        poses = json.load(f)

    return poses


def extract_pose(pose_dict):
    translation = np.array(pose_dict['cam_t_m2c'])
    rotation = np.array(pose_dict['cam_R_m2c']).reshape((3, 3))
    rotation = R.from_matrix(rotation).as_quat()

    pose_tgt = np.concatenate((rotation, translation))
    pose_src = alter_pose(pose_tgt)

    pose_deepim = np.zeros((1, 9))
    pose_deepim[0, 2:] = pose_tgt
    pose_tgt = pose_deepim.copy()

    pose_deepim = np.zeros((1, 9))
    pose_deepim[0, 2:] = pose_src
    pose_src = pose_deepim.copy()
    return pose_tgt, pose_src


def alter_pose(pose):
    rot = R.from_quat(pose[:4]).as_matrix()
    t = pose[4:]
    dr = R.from_euler('xyz', np.random.rand(3) * 0.2 - 0.1, degrees=False).as_matrix()
    rot = dr @ rot
    dt = np.random.rand(3) * avg_t * 0.1
    t = dt + t

    q = R.from_matrix(rot).as_quat()
    return np.concatenate((q, t))


def generate_samples(split='testing'):

    if split not in ['training', 'validation', 'testing']:
        print(f'invalid split name {split}')
        exit(-1)

    base_path = '/cvlabdata2/cvlab/datasets_protopap/swisscube/'

    images_list_path = os.path.join(base_path, f'{split}.txt')
    with open(images_list_path, 'r') as f:
        images_list = f.readlines()

    images = torch.FloatTensor(MINIBATCH_SIZE, 6, height, width).cuda().detach()
    flows = torch.FloatTensor(MINIBATCH_SIZE, 2, height, width).cuda().detach()
    imgs_tgt = torch.FloatTensor(MINIBATCH_SIZE, 3, height, width).cuda().detach()
    poses_src = np.zeros((MINIBATCH_SIZE, 9), dtype=np.float32)
    poses_tgt = np.zeros((MINIBATCH_SIZE, 9), dtype=np.float32)
    idx = 0
    for img_path in images_list:
        full_path = os.path.join('/cvlabdata2/home/yhu/data/SwissCube_1.0', img_path.strip())
        num = str(int(os.path.splitext(os.path.basename(full_path))[0]))

        seq_name = os.path.dirname(os.path.dirname(full_path))
        
        poses_name = os.path.join(seq_name, 'scene_gt.json')
        with open(poses_name, 'r') as j:
            poses = json.load(j)
        
        pose = poses[num][0]
        pose_tgt, pose_src = extract_pose(pose)
        poses_tgt[idx] = pose_tgt
        poses_src[idx] = pose_src

        img = cv2.imread(full_path)
        img = cv2.resize(img, (640, 640), cv2.INTER_AREA)
        img = img[80:560]

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))[np.newaxis, :] / 255  # N, C, H, W = (1, 3, 480, 640)
        images[idx, :3] = torch.from_numpy(img)

        idx += 1
        if idx == MINIBATCH_SIZE:
            idx = 0

            affine_matrices, zoom = process(poses_src, poses_tgt, images[:, 3:], imgs_tgt, flows)
            yield images, flows, poses_src, poses_tgt, affine_matrices, zoom


def train(gen_samples, network, optimizer, epoch):
    
    global weights_rot, extents, points
    total_batches = 30356 // MINIBATCH_SIZE
    total_pose, total_flow = 0.0, 0.0
    start = time.time()
    n_iter = 4
    renders = torch.FloatTensor(MINIBATCH_SIZE, 6, height, width)
    for curr_batch, sample in enumerate(gen_samples):
        images, flows, poses_src, poses_tgt, affine_matrices, zoom = sample

        for it in range(n_iter):
            # zoom in image
            grids = nn.functional.affine_grid(affine_matrices, images.size())
            input_zoom = nn.functional.grid_sample(images, grids).detach()

            # zoom in flow
            flow_zoom = nn.functional.grid_sample(flows, grids)
            for k in range(flow_zoom.shape[0]):
                flow_zoom[k, 0, :, :] /= affine_matrices[k, 0, 0] * 20.0
                flow_zoom[k, 1, :, :] /= affine_matrices[k, 1, 1] * 20.0

            output, loss_pose_tensor, quaternion_delta_var, translation_var = \
                network(input_zoom.float(), tweights_rot.float(),
                        torch.from_numpy(poses_src).cuda().detach(), torch.from_numpy(poses_tgt).cuda().detach(),
                        textents.float(), tpoints.float(), zoom.float())


            quaternion_delta = quaternion_delta_var.cpu().detach().numpy()
            translation = translation_var.cpu().detach().numpy()
            poses_est, error_rot, error_trans = \
                    _compute_pose_target(quaternion_delta, translation, poses_src, poses_tgt)

            # losses
            loss_pose = torch.mean(loss_pose_tensor)
            loss_flow = 0.1 * multiscaleEPE(output, flow_zoom)
            loss = loss_pose + loss_flow

            # compute gradient and do optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            end = time.time() - start
            start = time.time()
            total_pose += loss_pose.item()
            total_flow += loss_flow.item()

            print('batch: [%d/%d], iter %d, epoch: [%d/%d], loss %.4f, l_pose %.4f (r %.2f, t %.2f), l_flow %.4f, lr %.6f, in time %f'
                  % (curr_batch + 1, total_batches, it + 1, epoch, cfg.epochs, loss, loss_pose, error_rot, error_trans, loss_flow, loss_pose, end))

            affine_matrices, zoom = process(poses_est, poses_tgt, renders[:, :3], renders[:, 3:], flows)
            images[:, 3:].copy_(renders[:, :3])

    return total_pose / total_batches, total_flow / total_batches
                
                
if __name__ == '__main__':
    
    cfg_from_file('experiments/cfgs/swisscube.yml')

    cfg.renderer = Renderer(synthetic=True)
    network_path = 'data/checkpoints/from_deepim.pth'
    network = load_network(network_path)

    fine_tune = [param for name, param in network.module.named_parameters()]
    optimizer = torch.optim.SGD(fine_tune, cfg.TRAIN.LEARNING_RATE, momentum=cfg.TRAIN.MOMENTUM)
 

    pose_losses, flow_losses = [], []
    epochs = 25
    cfg.epochs = epochs
    for epoch in range(epochs):
        lpose, lflow = train(generate_samples('training'), network, optimizer, epoch)

        pose_losses.append(lpose)
        flow_losses.append(lflow)
        
        state = {'epoch': epoch + 1, 'state_dict': network.module.state_dict()}
        filename = 'data/checkpoints/swisscube/ours_epoch_{:d}'.format(epoch+1) + '_checkpoint.pth'
        torch.save(state, filename)

    np.save('losses', np.hstack((np.array(pose_losses)[:, np.newaxis], np.array(flow_losses))[:, np.newaxis]))
