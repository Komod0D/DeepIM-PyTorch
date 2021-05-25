#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

python3 -m pdb ./tools/estimate_swisscube.py --gpu 0 \
  --imgdir data/images/1b/ \
  --color *.bmp \
  --network flownets \
  --pretrained data/checkpoints/ycb_object/flownets_ycb_object_20objects_color_self_supervision_epoch_10.checkpoint.pth \
  --dataset swisscube_test \
  --cfg experiments/cfgs/swisscube.yml
