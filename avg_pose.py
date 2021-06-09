import os
import numpy as np
import json
from scipy.spatial.transform import Rotation as R

base_path = '/cvlabdata2/cvlab/datasets_protopap/swisscube/'
split = 'training'

def load_poses(path):
    with open(path, 'r') as f:
        poses = json.load(f)

    return poses

images_list_path = os.path.join(base_path, f'{split}.txt')
with open(images_list_path, 'r') as f:
    images_list = f.readlines()


def extract_pose(pose_dict):
    rotation = np.array(pose_dict['cam_R_m2c']).reshape((3, 3))
    translation = np.array(pose_dict['cam_t_m2c'])

    quaternion = R.from_matrix(rotation).as_quat()
    pose = np.concatenate((quaternion, translation))  # TODO: CHECK!!!!!!!!!!!!!!
    pose_deepim = np.zeros((1, 9))
    pose_deepim[0, 2:] = pose
    return pose


poses_tgt = []

for img_path in images_list[:100]:
    full_path = os.path.join(base_path, img_path.strip())
    pose_path = os.path.join(os.path.dirname(os.path.dirname(full_path)), 'scene_gt.json')
    poses = load_poses(pose_path)
    im_num = str(int(os.path.splitext(os.path.basename(full_path))[0]))
    pose_tgt = extract_pose(poses[im_num][0])  # TODO check again!!!!!!!
    poses_tgt.append(pose_tgt)


poses_tgt = np.vstack(poses_tgt)
print(np.mean(poses_tgt, axis=0))
