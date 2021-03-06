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
    --max_epoch 240 \
    --batch_size 100 
```
