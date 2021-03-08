"""
Copyright (C) 2021 Akram Sbaih, Stanford University
    You can contact the author at <akram at stanford dot edu>
This script helps you visualize the trajectories predicted against ground truth.
"""

import argparse
import scipy.io as io
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


def visualize(mat_file, out_dir, begin, end, increment, history_len):
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    mat_file = io.loadmat(mat_file)
    gt, pr = mat_file['gt'][..., :2], mat_file['pr'][..., :2]  # ignore the z coordinate
    if mat_file['input'] is not None and len(mat_file['input'].shape) > 1:
        history = mat_file['input'][..., :2]
        build_history = False
    else:
        history = gt[begin].reshape(-1, 2).copy().tolist()
        build_history = True
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    end = gt.shape[0] if end == -1 else end
    for i in tqdm(range(begin, end, increment)):
        ax.clear()
        # get correct slices
        gt_i, pr_i = gt[i].reshape(-1, 2).T, pr[i].reshape(-1, 2).T
        hs_i = np.array(history).T if build_history else history[i].reshape(-1, 2).T
        # center them on current location
        zero = gt_i[:, 0:1].copy()
        gt_i -= zero
        pr_i -= zero
        hs_i -= zero
        # contain them all on the same plot by finding the maximum distance from zero
        max_dis = np.max([np.max(np.abs(gt_i)), np.max(np.abs(gt_i)), np.max(np.abs(hs_i))]) * 0.7
        ax.set_xlim([-max_dis, max_dis])
        ax.set_ylim([-max_dis, max_dis])
        # and plot them all
        ax.plot(*gt_i, 'D-.g', label='GT')
        ax.plot(*pr_i, 'D-.b', label='Predicted')
        ax.plot(*hs_i, 'D-.m', label='History')
        ax.set_title(f"{i:04}/{end:04}.png")
        ax.legend()
        # update history
        if build_history:
            history.append(zero.reshape(1, 2).tolist()[0])
            history = history[-history_len:]
        # save figure
        fig.savefig(os.path.join(out_dir, f"{i:04}.png"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize trajectories.')
    parser.add_argument('--mat_file', type=str, default='output/Individual/baseline/det_0.mat',
                        help="File generated from the model containing predictions")
    parser.add_argument('--out_dir', type=str, default='vis',
                        help="Where to store the generated images")
    parser.add_argument('--begin', type=int, default=0,
                        help="Which frame to begin with")
    parser.add_argument('--end', type=int, default=-1,
                        help="Which frame to end with; -1 means end")
    parser.add_argument('--inc', type=int, default=1,
                        help="How many frames to skip every image")
    parser.add_argument('--history_len', type=int, default=12,
                        help="Number of frames to keep as the brown history")
    args = parser.parse_args()
    visualize(args.mat_file, args.out_dir, args.begin, args.end, args.inc, args.history_len)
