# Detailed Analysis of Graph Neural Networks

## Overview

This repository contains experiments on various tasks, namely - multi class node classification (nc), link prediction (lp), and pairwise node classification (pnc) (motivated from [P-GNN](https://arxiv.org/abs/1906.04817)), using the Graph Attention Network architecture.

## Setup

- Install PyTorch from the official website. We are using 1.4.0 version for our experiments. Replace 10.2 with your cuda version.
```bash
$ conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
```

- Install the [pyTorch geometric](https://github.com/rusty1s/pytorch_geometric) library using the following commands. Replace ${CUDA} with your CUDA version (refer to the link for more details).
```bash
$ pip install torch-scatter==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.4.0.html
$ pip install torch-sparse==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.4.0.html
$ pip install torch-spline-conv==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.4.0.html
$ pip install torch-cluster (optional)
$ pip install torch-geometric
```

- Install dgl, networkx and tensorboardX libraries. Replace cu102 with your CUDA version.
```bash
$ pip install dgl-cu102 networkx tensorboardX
```

## Datasets

We are working with four datasets:

- Protein Protein Interaction (PPI)
- Proteins
- Brightkite
- Communities

They can be downloaded by running the `get_data.sh` script. The communities dataset is loaded by the networkx library.

## References

The baseline code was forked from the following repositories:

- Node Classification: [dgl](https://github.com/dmlc/dgl)
- Link Prediction & Pairwise Node Classification: [P-GNN](https://github.com/JiaxuanYou/P-GNN)