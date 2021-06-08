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

        self.renderer = pyrender.OffscreenRenderer(viewport_width=1024, viewport_height=1024, point_size=1.0)
        scene = pyrender.Scene(ambient_light=[0.02, 0.02, 0.02], bg_color=[0, 0, 0])

        light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=1000000.0)
        cam = pyrender.IntrinsicsCamera(607.57, 607.57, 512, 512, zfar=2000)
        cam_rot = R.from_euler('y', 180, degrees=True).as_matrix()
        cam_matrix = to_homo(cam_rot, np.zeros((3,)))

        self.nm = pyrender.Node(mesh=mesh, matrix=np.eye(4))
        nl = pyrender.Node(light=light, matrix=np.eye(4))
        nc = pyrender.Node(camera=cam, matrix=cam_matrix)

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
        translation, rotation_quat = pose[:3], pose[3:]
        translation = np.array(translation)

        rotation = R.from_quat(rotation_quat).as_matrix()
        transform = to_homo(rotation, translation)
        self.scene.set_pose(self.nm, pose=transform)

    def render_(self):
        color, depth = self.renderer.render(self.scene)
        """
        color = cv2.resize(color, (640, 640), cv2.INTER_AREA)
        color = color[80:560]
        """
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


def get_next(iteritems):
    """"
    img, pose = next(iteritems)
    img = os.path.join(img.split('/')[0], 'Test', img.split('/')[1])
    img = cv2.imread(img)
    img = cv2.resize(img, (640, 640), cv2.INTER_AREA)
    img = img[80:560]
    translation = pose['t']
    rotation = pose['r']
    """

    img_path = next(iteritems)
    full_path = os.path.join('/cvlabdata2/home/yhu/data/SwissCube_1.0', img_path)
    num = str(int(os.path.splitext(os.path.basename(full_path))[0]))
    img = cv2.imread(full_path)

    seq_name = os.path.split(full_path)[:-2]
    poses_name = os.path.join(*seq_name, 'scene_gt.json')
    with open(poses_name, 'r') as j:
        poses = json.load(j)

    pose = poses[num]
    translation = np.array(pose['cam_t_m2c'])
    rotation = np.array(pose['cam_R_m2c']).reshape((3, 3))

    return img, translation, R.from_matrix(rotation).as_quat()


if __name__ == '__main__':
    r = Renderer()

    import os
    import json

    os.chdir('/cvlabdata2/home/yhu/data/SwissCube_1.0')
    with open('testing.txt', 'r') as f:
        images = f.readlines()

    iteritems = iter(images)
    img, translation, rotation = get_next(iteritems)

    cv2.imshow('image', img)
    x, y, z = translation

    while True:
        a, b, c, d = rotation
        pose = [x, y, z, a, b, c, d]
        r.set_pose(pose)
        print(R.from_quat(rotation).as_euler('xyz', degrees=True))
        color = r.render_()

        cv2.imshow('render', color)
        key = cv2.waitKey(0)
        if key == 27:
            break
        elif key == 13:
            img, translation, rotation = get_next(iteritems)
            x, y, z = translation
            a, b, c, d = rotation

            cv2.imshow('image', img)
        elif key == 119:
            y += 10
        elif key == 115:
            y -= 10
        elif key == 97:
            x -= 10
        elif key == 100:
            x += 10
        elif key == 101:
            z += 10
        elif key == 113:
            z -= 10
        elif key == 81:
            rotation = R.from_euler('y', -15, degrees=True).as_matrix() @ R.from_quat(rotation).as_matrix()
            rotation = R.from_matrix(rotation).as_quat()
        elif key == 83:
            rotation = R.from_euler('y', 15, degrees=True).as_matrix() @ R.from_quat(rotation).as_matrix()
            rotation = R.from_matrix(rotation).as_quat()
        elif key == 82:
            rotation = R.from_euler('x', -15, degrees=True).as_matrix() @ R.from_quat(rotation).as_matrix()
            rotation = R.from_matrix(rotation).as_quat()
        elif key == 84:
            rotation = R.from_euler('x', 15, degrees=True).as_matrix() @ R.from_quat(rotation).as_matrix()
            rotation = R.from_matrix(rotation).as_quat()
