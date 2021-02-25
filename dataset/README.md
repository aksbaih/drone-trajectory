## How to get the data
This directory helps you download and prepare the [UZH FPV Dataset](https://fpv.ifi.uzh.ch/datasets/).
* `dataset_links.txt` are the links to download all the Leica zip files listed [here](https://fpv.ifi.uzh.ch/datasets/) as of Feb 24, 2021.
* `download_preprocess.py` takes the previous list, downloads it, unzips it, and runs a script adapted from [the dataseet toolkit](https://github.com/uzh-rpg/uzh_fpv_open) which converts it to the following format and stores it in `data/` directory.

## Data format
After following the previous steps, you'll have the following structure:
```
data -
     |- t   - 
            |- <env name>_<id>.npy
            |- <env name>_<id>.npy
            ...
     |- xyz - 
            |- <env name>_<id>.npy
            |- <env name>_<id>.npy
            ...
```
For each file, `<env name>` and `<env id>` match the naming in the original dataset and they describe the different data collection runs performed in different environments and different routes and velocities. The files under `t` are the timestamps. For each of them, there's a file with the same name under `xyz` and they map one-to-one to the absolute location of the drone at that timestamp.  