import pyrender
import os
import trimesh
import numpy as np
import cv2
import os

from scipy.spatial.transform import Rotation as R

os.environ['PYOPENGL_PLATFORM'] = 'egl'


def to_homo(rotation, translation):
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    
    return transform


class Renderer():
    def __init__(self):
        os.environ['PYOPENGL_PLATFORM'] = 'egl'
        tscene = trimesh.load('/cvlabdata2/cvlab/datasets_protopap/deepim/data/models/swisscube/swisscube.obj')
        mesh = pyrender.Mesh.from_trimesh(list(tscene.geometry.values()), smooth=False)

        self.r = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480, point_size=1.0)
        scene = pyrender.Scene(ambient_light=[0.02, 0.02, 0.02], bg_color=[0, 0, 0])

        light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=100.0)
        cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.414)
        self.nm = pyrender.Node(mesh=mesh, matrix=np.eye(4))
        nl = pyrender.Node(light=light, matrix=np.eye(4))
        nc = pyrender.Node(camera=cam, matrix=np.eye(4))

        scene.add_node(self.nm)
        scene.add_node(nl)
        scene.add_node(nc)
        
        self.scene = scene
        



    def set_pose(self, pose):
        rotation_quat, translation = pose[:4], pose[4:]
        rotation = R.from_quat(rotation_quat).as_matrix()
        transform = to_homo(rotation, translation)
        self.scene.set_pose(self.nm, pose=transform)


    def render(self):
        color, depth = self.r.render(self.scene)
        return color
    

if __name__ == '__main__':
    r = Renderer()


    x, y, z = 0, 0, 0

    while (True):
        r.set_pose([0, 0, 0, 1, x, y, z])

        color = r.render()

        cv2.imshow('render', color)
        key = cv2.waitKey(0)
        if key == 27:
            break
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
