## Baseline Transformer Architecture

I'm using the [Trajectory-Transformer](github.com/FGiuliari/Trajectory-Transformer.git) repo as the baseline model as described in [my report](../report). Make sure the git submodule is initialized as a subdirectory [here](Trajectory-Transformer) as described in the [main README](../README.md). If not, run the following command
```
git submodule init
git submodule update
```

## Applying the modifications
You need to copy the modified files by running the command
```
sh modifications/apply_modifications.sh
```

## Training the baseline

```
cd Trajectory-Transformer
CUDA_VISIBLE_DEVICES=0 python train_individualTF.py \
    --dataset_folder ../../dataset \
    --dataset_name data \
    --name baseline \
    --obs 12 --preds 8 \
    --val_size 64 \
    --max_epoch 360 \
    --save_step 5 \
    --eval_every 5 \
    --batch_size 64 
```

## Visualizing the predictions
You can visualize the generations on the test set at any epoch using [this script](modifications/visual_utils.py) by running the following command
```
cd Trajectory-Transformer
python visual_utils.py \
    --mat_file output/Individual/baseline/det_148.mat \
    --out_dir vis \
    --begin 400 \
    --end 500 \
```
Look at the script to see what other arguments you can change. The `--mat_file` is generated during training of each epoch in the training output directory. 

You can convert the sequence of images generated above into a movie using
```
sudo apt install ffmpeg
cd vis
ffmpeg -r 8 -f image2 -s 1080x1080 -i %04d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p vis.mp4
```
