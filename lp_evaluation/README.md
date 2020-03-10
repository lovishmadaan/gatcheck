# Link Prediction Analysis


## Run
- 3-layer GAT, PPI, link
```bash
bash run.sh
```

Complete list of parameters is present in `args.py` and you can make the changes in `run.sh` to do various experiments for grids and communities datasets.

We recommend using tensorboard to monitor the training process. To do this, you may run
```bash
tensorboard --logdir runs
```