#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

python3 -m pdb ./estimate_linemod.py --gpu 0 \
  --imgdir data/images/linemod/000001/rgb/ \
  --color *.png \
  --network flownets \
  --pretrained data/checkpoints/ycb_object/flownets_ycb_object_20objects_color_self_supervision_epoch_10.checkpoint.pth \
  --dataset linemod_test \
  --cfg experiments/cfgs/swisscube.yml
