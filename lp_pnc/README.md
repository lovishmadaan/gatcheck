# Link Prediction and Pairwise Node Classification


## Run
- 3-layer GAT, PPI, link
```bash
python -u main.py --model GAT --layer_num 3 --dataset ppi --task link |& tee out_ppi.txt
```
- 3-layer GAT, Proteins, link_pair
```bash
python -u main.py --model GAT --layer_num 3 --dataset proteins --task link_pair |& tee out_proteins.txt
```

Complete list of parameters is present in `args.py`

We recommend using tensorboard to monitor the training process. To do this, you may run
```bash
tensorboard --logdir runs
```