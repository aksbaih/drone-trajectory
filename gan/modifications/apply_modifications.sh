#!/usr/bin/env bash

# First copy the modifications from the baseline
cp ../baseline/modifications/baselineUtils.py Trajectory-Transformer
cp ../baseline/modifications/train_individualTF.py Trajectory-Transformer
cp ../baseline/modifications/visual_utils.py Trajectory-Transformer

# Now copy the new modifications for GAN
cp modifications/train_gan.py Trajectory-Transformer
cp modifications/gan.py Trajectory-Transformer
