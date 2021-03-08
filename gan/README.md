## Novel GAN-based Approach

...

## Backbone Transformer Architecture
This section is to recreate the baseline here and then build off it.

I'm using the [Trajectory-Transformer](github.com/FGiuliari/Trajectory-Transformer.git) repo as the baseline model as described in [my report](../report). Make sure the git submodule is initialized as a subdirectory [here](Trajectory-Transformer) as described in the [main README](../README.md). If not, run the following command
```
git submodule init
git submodule update
```

## Applying the modifications
You need to copy the modified files by running the command (this also copies the modifications in [the baseline](../baseline))
```
sh modifications/apply_modifications.sh
```
