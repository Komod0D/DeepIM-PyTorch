from bop_toolkit_lib import renderer

from scipy.spatial.transform import Rotation as R
import numpy as np
import torch
import cv2

from bop_toolkit_lib import config
from bop_toolkit_lib import dataset_params
from bop_toolkit_lib import inout
from bop_toolkit_lib import misc
from bop_toolkit_lib import renderer
from bop_toolkit_lib import view_sampler
import os


def pose_to_view(pose):
    x, y, z, a, b, c, d = pose
    translation = np.array([x, y, z])
    rotation = np.array([a, b, c, d])
    rot_quat = R.from_quat(rotation)
    rot_mat = rot_quat.as_matrix()
    return rot_mat, translation


class RendererAdapter:
    def __init__(self, width, height):
        self.width, self.height = width, height
        self.renderer = renderer.create_renderer(width, height, 'python', mode='rgb', shading='phong')
        self.renderer.set_light_ambient_weight(0.5)


    def load_object(self, obj_id):
        self.renderer.add_object(obj_id, f'/cvlabdata2/cvlab/datasets_protopap/linemod/base/lm/models/obj_{obj_id:06d}.ply')


    def set_light_pos(self, *args):
        pass

    def set_light_color(self, *args):
        pass


    def set_projection_matrix(self, width, height, fx, fy, px, py, znear, zfar):
        self.fx = fx
        self.fy = fy
        self.cx = px
        self.cy = py

    
    def set_poses(self, poses):
        self.poses = poses
    
    def render_object(self, obj_id, rotation, translation, fx, fy, cx, cy):
        return self.renderer.render_object(obj_id, rotation, translation, fx, fy, cx, cy)

    def render(self, cls_indices, image_tensor, seg_tensor, pc2_tensor=None):
        R, t = pose_to_view(self.poses[0])

        rgb = self.renderer.render_object(cls_indices[0] + 1, R, t, self.fx, self.fy, self.cx, self.cy)['rgb']
        rgb = cv2.resize(rgb, (self.width, self.height), interpolation=cv2.INTER_AREA)
        temp = np.ones((rgb.shape[0], rgb.shape[1], rgb.shape[2] + 1))
        temp[:, :, :3] = rgb / 255
        tensor = torch.from_numpy(temp)
        image_tensor.copy_(tensor.flip(0))

        seg = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        seg = np.where(seg > 0, 0.25, 0)
        temp = np.zeros((rgb.shape[0], rgb.shape[1], rgb.shape[2] + 1))
        temp[:, :, 2] = seg[:, :]
        temp[:, :, 3] = np.ones((rgb.shape[0], rgb.shape[1]))
        seg_t = torch.from_numpy(temp)
        seg_tensor.copy_(seg_t.flip(0))

        if (pc2_tensor is not None):
            pc2_tensor.copy_(image_tensor)
