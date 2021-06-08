import pyrender
import os
import trimesh
import numpy as np
import cv2
import os
import torch

from scipy.spatial.transform import Rotation as R

os.environ['PYOPENGL_PLATFORM'] = 'egl'


def to_homo(rotation, translation):
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    
    return transform


class Renderer:
    def __init__(self):
        os.environ['PYOPENGL_PLATFORM'] = 'egl'
        tscene = trimesh.load('/cvlabdata2/cvlab/datasets_protopap/deepim/data/models/swisscube/swisscube.obj')
        mesh = pyrender.Mesh.from_trimesh(list(tscene.geometry.values()), smooth=False)

        self.renderer = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480, point_size=1.0)
        scene = pyrender.Scene(ambient_light=[0.02, 0.02, 0.02], bg_color=[0, 0, 0])

        light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=100.0)
        cam = pyrender.IntrinsicsCamera(4000, 4000, 1024, 1024, zfar=1000)
        self.nm = pyrender.Node(mesh=mesh, matrix=np.eye(4))
        nl = pyrender.Node(light=light, matrix=np.eye(4))
        nc = pyrender.Node(camera=cam, matrix=np.eye(4))

        scene.add_node(self.nm)
        scene.add_node(nl)
        scene.add_node(nc)
        
        self.scene = scene

    def set_light_pos(self, *args):
        pass

    def set_light_color(self, *args):
        pass

    def set_projection_matrix(self, width, height, fx, fy, px, py, znear, zfar):
        pass

    def set_poses(self, poses):
        self.set_pose(poses[0])

    def set_pose(self, pose):
        rotation_quat, translation = pose[:4], pose[4:]
        rotation = R.from_quat(rotation_quat).as_matrix()
        transform = to_homo(rotation, translation)
        self.scene.set_pose(self.nm, pose=transform)

    def render_(self):
        color, depth = self.renderer.render(self.scene)
        return color

    def render(self, cls_indices, image_tensor, seg_tensor, pc2_tensor=None):
        rgb = self.render_()

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
