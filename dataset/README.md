## How to get the data
This directory helps you download and prepare the [UZH FPV Dataset](https://fpv.ifi.uzh.ch/datasets/).
* `dataset_links.txt` are the links to download all the Leica zip files listed [here](https://fpv.ifi.uzh.ch/datasets/) as of Feb 24, 2021.
* `download_preprocess.py` takes the previous list, downloads it, unzips it, and formats it to a provided `FPS` and stores it locally.
* `uzh_fpv_utils.py` is a script adapted from [the dataseet toolkit](https://github.com/uzh-rpg/uzh_fpv_open) to parse `Leica` format.

## Data format
After following the previous steps, you'll have the following structure:
```
data -
     |-train- 
            |- <env name>_<id>.txt
            |- <env name>_<id>.txt
            ...
```
For each file, `<env name>` and `<env id>` match the naming in the original dataset and they describe the different data collection runs performed in different environments and different routes and velocities. 

The file follows the dataset format for the baseline [Trajectory Transformer](https://github.com/FGiuliari/Trajectory-Transformer). Each row is a frame in a constant `FPS` chosen at [download_preprocess.py](download_preprocess.py). The columns are `frame_idx,drone_idx,x,y,z` and are tab-separated. 

If you wish to have a test set, you can move some of the data files to a separate test set directory. 