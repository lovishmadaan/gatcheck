"""
Graph Attention Networks (PPI Dataset) in DGL using SPMV optimization.
Multiple heads are also batched together for faster training.
Compared with the original paper, this code implements
early stopping.
References
----------
Paper: https://arxiv.org/abs/1710.10903
Author's code: https://github.com/PetarV-/GAT
Pytorch implementation: https://github.com/Diego999/pyGAT
"""
import random
import inspect
import copy
import numpy as np
import torch
import dgl
import networkx as nx
import torch.nn.functional as F
import argparse
from sklearn.metrics import precision_score, recall_score, f1_score
from gat import GAT
from dgl.data.ppi import LegacyPPIDataset
from torch.utils.data import DataLoader

def collate(sample):
    graphs, feats, labels =map(list, zip(*sample))
    graph = dgl.batch(graphs)
    feats = torch.from_numpy(np.concatenate(feats))
    labels = torch.from_numpy(np.concatenate(labels))
    return graph, feats, labels

def evaluate(feats, model, subgraph, labels, loss_fcn):
    with torch.no_grad():
        model.eval()
        model.g = subgraph
        for layer in model.gat_layers:
            layer.g = subgraph
        output = model(feats.float())
        loss_data = loss_fcn(output, labels.float())
        predict = np.where(output.data.cpu().numpy() >= 0.5, 1, 0)
        precision = precision_score(labels.data.cpu().numpy(),
                         predict, average='micro')
        recall = recall_score(labels.data.cpu().numpy(),
                         predict, average='micro')
        score = f1_score(labels.data.cpu().numpy(),
                         predict, average='micro')
        return precision, recall, score, loss_data.item()

def modify(curr_dataset, train_dataset, curr_list, mode='train', offset=0):
    tmp_curr_graphs = [train_dataset.train_graphs[i] for i in curr_list]
    tmp_mask_list = [train_dataset.train_mask_list[i] for i in curr_list]

    tmp_graph = dgl.batch(tmp_curr_graphs)
    node_num_list = tmp_graph.batch_num_nodes

    lst = []
    for index, val in enumerate(node_num_list):
        lst = lst + list([offset + index + 1] * val)
    tmp_graph_id = np.array([lst])

    ind_list = tmp_mask_list[0]
    for i in range(1, len(tmp_mask_list)):
        ind_list = np.concatenate((ind_list, tmp_mask_list[i]), axis=None)
    
    tmp_labels = train_dataset.labels[ind_list, :]
    tmp_features = train_dataset.features[ind_list, :]

    curr_nodes = node_num_list[0]
    tmp_curr_mask_list = [np.array(list(range(curr_nodes)))]
    tmp_curr_labels = [tmp_labels[tmp_curr_mask_list[0], :]]
    for i in range(1, len(node_num_list)):
        curr = np.array(list(range(curr_nodes, curr_nodes + node_num_list[i])))
        curr_nodes += node_num_list[i]
        tmp_curr_mask_list.append(curr)
        tmp_curr_labels.append(tmp_labels[curr, :])

    curr_dataset.labels = tmp_labels
    curr_dataset.features = tmp_features
    curr_dataset.graph_id = tmp_graph_id
    curr_dataset.graph = tmp_graph
    if mode == 'valid':
        curr_dataset.valid_mask_list = tmp_curr_mask_list
        curr_dataset.valid_graphs = tmp_curr_graphs
        curr_dataset.valid_labels = tmp_curr_labels
    else:
        curr_dataset.train_mask_list = tmp_curr_mask_list
        curr_dataset.train_graphs = tmp_curr_graphs
        curr_dataset.train_labels = tmp_curr_labels
    
    return


        
def main(args):
    if args.gpu<0:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:" + str(args.gpu))

    # batch_size = args.batch_size
    # cur_step = 0
    # patience = args.patience
    # best_score = -1
    # best_loss = 10000
    # # define loss function
    # loss_fcn = torch.nn.BCEWithLogitsLoss()

    # create the dataset
    train_dataset = LegacyPPIDataset(mode='train')
    valid_dataset = LegacyPPIDataset(mode='valid')
    test_dataset = LegacyPPIDataset(mode='test')

    # nxg = valid_dataset.graph.to_networkx().to_undirected()
    # comps = [comp for comp in nx.connected_components(nxg) if len(comp)>10]
    # print(len(comps))
    # exit()

    cross_valid_list = []
    for i in range(5):
        cross_valid_list.append(list(range(4*i, 4*(i + 1))))
    cross_train_dataset = copy.copy(train_dataset)

    valid_precision = []
    valid_recall = []
    valid_scores = []
    test_precision = []
    test_recall = []
    test_scores = []
    for ind, valid_list in enumerate(cross_valid_list):
        batch_size = args.batch_size
        cur_step = 0
        patience = args.patience
        best_score = -1
        best_loss = 10000
        # define loss function
        loss_fcn = torch.nn.BCEWithLogitsLoss()

        train_list = [ind for ind in range(20) if ind not in valid_list]
        print('Train List: {}'.format(train_list))
        print('Valid List: {}'.format(valid_list))
        modify(train_dataset, cross_train_dataset, train_list, mode='train', offset=0)
        modify(valid_dataset, cross_train_dataset, valid_list, mode='valid', offset=16)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collate)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate)
        n_classes = train_dataset.labels.shape[1]
        num_feats = train_dataset.features.shape[1]
        g = train_dataset.graph
        heads = ([args.num_heads] * args.num_layers) + [args.num_out_heads]
        # define the model
        model = GAT(g,
                    args.num_layers,
                    num_feats,
                    args.num_hidden,
                    n_classes,
                    heads,
                    F.elu,
                    args.in_drop,
                    args.attn_drop,
                    args.alpha,
                    args.residual)
        # define the optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        model = model.to(device)

        for epoch in range(args.epochs):
            model.train()
            loss_list = []
            for batch, data in enumerate(train_dataloader):
                subgraph, feats, labels = data
                feats = feats.to(device)
                labels = labels.to(device)
                model.g = subgraph
                for layer in model.gat_layers:
                    layer.g = subgraph
                logits = model(feats.float())
                loss = loss_fcn(logits, labels.float())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_list.append(loss.item())
            loss_data = np.array(loss_list).mean()
            print("Epoch {:05d} | Loss: {:.4f}".format(epoch + 1, loss_data), end=' ')
            if epoch % 1 == 0:
                score_list = []
                val_loss_list = []
                for batch, valid_data in enumerate(valid_dataloader):
                    subgraph, feats, labels = valid_data
                    feats = feats.to(device)
                    labels = labels.to(device)
                    prec, recall, score, val_loss = evaluate(feats.float(), model, subgraph, labels.float(), loss_fcn)
                    score_list.append([prec, recall, score])
                    val_loss_list.append(val_loss)
                mean_score = np.array(score_list).mean(axis=0)
                mean_val_loss = np.array(val_loss_list).mean()
                print("| Valid Precision: {:.4f} | Valid Recall: {:.4f} |  Valid F1-Score: {:.4f} ".format(mean_score[0], mean_score[1], mean_score[2]), end = ' ')

                test_score_list = []
                for batch, test_data in enumerate(test_dataloader):
                    subgraph, feats, labels = test_data
                    feats = feats.to(device)
                    labels = labels.to(device)
                    test_prec, test_rec, test_score, _ = evaluate(feats, model, subgraph, labels.float(), loss_fcn)
                    test_score_list.append([test_prec, test_rec, test_score])
                mean_test_score = np.array(test_score_list).mean(axis=0)
                print("| Test Precision: {:.4f} | Test Recall: {:.4f} | Test F1-Score: {:.4f}".format(mean_test_score[0], mean_test_score[1], mean_test_score[2]))

                if epoch == args.epochs - 1:
                    valid_precision.append(round(mean_score[0], 4))
                    valid_recall.append(round(mean_score[1], 4))
                    valid_scores.append(round(mean_score[2], 4))
                    test_precision.append(round(mean_test_score[0], 4))
                    test_recall.append(round(mean_test_score[1], 4))
                    test_scores.append(round(mean_test_score[2], 4))

                # early stop
                if mean_score[2] > best_score or best_loss > mean_val_loss:
                    if mean_score[2] > best_score and best_loss > mean_val_loss:
                        val_early_loss = mean_val_loss
                        val_early_score = mean_score[2]
                    best_score = np.max((mean_score[2], best_score))
                    best_loss = np.min((best_loss, mean_val_loss))
                    cur_step = 0
                else:
                    cur_step += 1
                    if cur_step == patience:
                        valid_precision.append(round(mean_score[0], 4))
                        valid_recall.append(round(mean_score[1], 4))
                        valid_scores.append(round(mean_score[2], 4))
                        test_precision.append(round(mean_test_score[0], 4))
                        test_recall.append(round(mean_test_score[1], 4))
                        test_scores.append(round(mean_test_score[2], 4))
                        break
        print('Valid Scores: {}'.format(valid_scores))
        print('Test Scores: {}'.format(test_scores))
    
    out_matrix = np.stack([valid_precision, valid_recall, valid_scores, test_precision, test_recall, test_scores], axis=1)
    np.savetxt('results.csv', out_matrix, delimiter=',')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GAT')
    parser.add_argument("--gpu", type=int, default=-1,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--epochs", type=int, default=400,
                        help="number of training epochs")
    parser.add_argument("--num-heads", type=int, default=4,
                        help="number of hidden attention heads")
    parser.add_argument("--num-out-heads", type=int, default=6,
                        help="number of output attention heads")
    parser.add_argument("--num-layers", type=int, default=2,
                        help="number of hidden layers")
    parser.add_argument("--num-hidden", type=int, default=256,
                        help="number of hidden units")
    parser.add_argument("--residual", action="store_true", default=True,
                        help="use residual connection")
    parser.add_argument("--in-drop", type=float, default=0,
                        help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=0,
                        help="attention dropout")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=0,
                        help="weight decay")
    parser.add_argument('--alpha', type=float, default=0.2,
                        help="the negative slop of leaky relu")
    parser.add_argument('--batch-size', type=int, default=2,
                        help="batch size used for training, validation and test")
    parser.add_argument('--patience', type=int, default=10,
                        help="used for early stop")
    args = parser.parse_args()
    print(args)

    main(args)
