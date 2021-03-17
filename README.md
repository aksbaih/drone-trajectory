# drone-trajectory
Project for Stanford CS236G involving transformers and GANs. Refer to [the report](report/report.pdf) for more.

The task at hand is to predict multiple possible trajectories of a drone for the next second given observations of its location for the past 1.5 seconds.

* [./baseline/](baseline) shows the implementation and instructions to train and run the baseline transformer.
* [./gan/](gan) shows the implementation and instructions to train and run my new 3 approaches for a GAN-Transformer-based model to predict a variety of trajectories.

## Submodules
This repo contains the following submodule repos:
* baseline [Trajectory Transformer](https://github.com/FGiuliari/Trajectory-Transformer) used [here](baseline).

To clone correctly, use the following command
```
git clone --recurse-submodules https://github.com/aksbaih/drone-trajectory
```

## Future Thoughts
GAN's are not the best approach for this task as there are different architectures such as GPT which proved good performance in similar tasks in NLP and require less training.

