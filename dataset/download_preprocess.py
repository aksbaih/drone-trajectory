"""
Copyright (C) 2021 Akram Sbaih, Stanford University
    You can contact the author at <akram at stanford dot edu>

This script helps you download and process the UZH FPV dataset
described at https://github.com/uzh-rpg/uzh_fpv_open
"""

import numpy as np
import pandas as pd
import os
import wget
import zipfile
from tqdm import tqdm
from .uzh_fpv_utils import parseTextFile

if __name__=="__main__":
    # download the zip files listed in dataset_links.txt
    # if you already have them downloaded, you can comment out the following lines
    print("Downloading zips...")
    links = [line.strip() for line in open("dataset_links.txt", 'r').readlines() if line and line[0] != '#']
    zips = []
    if not os.path.exists('zips'): os.mkdir('zips')
    for link in tqdm(links):
        zips.append("zips/" + link.split('/')[-1])
        wget.download(link, zips[-1])
    zips = ['zips/'+file for file in os.listdir('zips') if file.endswith('.zip')]
    # unzip files
    print("Unzipping...")
    unzips = []
    if not os.path.exists('unzips'): os.mkdir('unzips')
    for zip in tqdm(zips):
        unzips.append(zip.replace('zips/', 'unzips/').replace('.zip', ''))
        with zipfile.ZipFile(zip, 'r') as zip_ref:
            zip_ref.extractall('unzips/')
    # parse files and store as np files
    print("Preprocessing...")
    if not os.path.exists('data'): os.mkdir('data')
    if not os.path.exists('data/train'): os.mkdir('data/train')
    for unzip in tqdm(unzips):
        identifier = unzip.split('/')[-1]
        timestamps, xyz = parseTextFile(unzip+'/leica.txt')
        # start timestamps at 0 instead
        timestamps = np.array([ts.to_sec() for ts in timestamps]) - timestamps[0].to_sec()
        file_duration = timestamps[-1]
        # new timestamps that follow a constant FPS
        FPS = 8
        desired_timestamps = np.linspace(0, file_duration, int(FPS * file_duration))
        # transform to constant FPS and then to integer frame index
        df = pd.DataFrame(xyz, index=timestamps)
        df = df[~df.index.duplicated(keep='first')]\
            .reindex(desired_timestamps, method='nearest')
        df.index = list(range(int(file_duration * FPS)))
        # insert a pedastrian column with a constant value... this could be used to reflect having multiple drones
        df.insert(0, "ped", 1.)
        df.to_csv("data/train/"+identifier+".txt", sep="\t", header=False)
    print("Done!")
