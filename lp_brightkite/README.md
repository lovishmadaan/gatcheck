# Link Prediction and Pairwise Node Classification


## Run
- 3-layer GAT, PPI, link
```bash
python -u main.py --model GAT --layer_num 3 --dataset brightkite --task link |& tee out_brightkite.txt
```

Complete list of parameters is present in `args.py`

We recommend using tensorboard to monitor the training process. To do this, you may run
```bash
tensorboard --logdir runs
```