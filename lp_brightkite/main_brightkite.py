from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, f1_score
from tensorboardX import SummaryWriter

from args import *
from model import *
from utils import *
from dataset import *
import numpy as np

if not os.path.isdir('results'):
	os.mkdir('results')
# args
args = make_args()
print(args)
np.random.seed(123)
np.random.seed()
writer_train = SummaryWriter(comment=args.task+'_'+args.model+'_'+args.comment+'_train')
writer_val = SummaryWriter(comment=args.task+'_'+args.model+'_'+args.comment+'_val')
writer_test = SummaryWriter(comment=args.task+'_'+args.model+'_'+args.comment+'_test')


# set up gpu
if args.gpu:
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
	print('Using GPU {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))
else:
	print('Using CPU')
device = torch.device('cuda:'+str(args.cuda) if args.gpu else 'cpu')


# for task in ['link', 'link_pair']:
#     args.task = task
if args.dataset=='All':
	if task == 'link':
		datasets_name = ['grid','communities','ppi']
	else:
		datasets_name = ['communities', 'email', 'protein']
else:
	datasets_name = [args.dataset]
for dataset_name in datasets_name:
	# if dataset_name in ['communities','grid']:
	#     args.cache = False
	# else:
	#     args.epoch_num = 401
	#     args.cache = True
	results = []
	for repeat in range(1):
		time1 = time.time()
		data_list = get_tg_dataset(args, dataset_name, use_cache=args.cache, remove_feature=args.rm_feature)
		time2 = time.time()
		print(dataset_name, 'load time',  time2-time1)
		for k in range(5):
			result_val = []
			result_test = []
			num_features = data_list[0][k].x.shape[1]
			num_node_classes = None
			num_graph_classes = None
			if 'y' in data_list[0][k].__dict__ and data_list[0][k].y is not None:
				num_node_classes = max([data[k].y.max().item() for data in data_list])+1
			if 'y_graph' in data_list[0][k].__dict__ and data_list[0][k].y_graph is not None:
				num_graph_classes = max([data[k].y_graph.numpy()[0] for data in data_list])+1
			print('Dataset', dataset_name, 'Graph', len(data_list), 'Feature', num_features, 'Node Class', num_node_classes, 'Graph Class', num_graph_classes)
			nodes = [data[k].num_nodes for data in data_list]
			# for data in dat
			# print()
			edges = [data[k].num_edges for data in data_list]
			print('Node: max{}, min{}, mean{}'.format(max(nodes), min(nodes), sum(nodes)/len(nodes)))
			print('Edge: max{}, min{}, mean{}'.format(max(edges), min(edges), sum(edges)/len(edges)))

			args.batch_size = min(args.batch_size, len(data_list))
			print('Anchor num {}, Batch size {}'.format(args.anchor_num, args.batch_size))

			# data
			for i,data in enumerate(data_list):
				preselect_anchor(data[k], layer_num=args.layer_num, anchor_num=args.anchor_num, device='cpu')
				# data[k] = data[k].to(device)
				data_list[i][k] = data[k]

			# model
			input_dim = num_features
			output_dim = args.output_dim
			model = locals()[args.model](input_dim=input_dim, attention_heads=args.attention_heads, out_attention_heads=args.out_attention_heads, feature_dim=args.feature_dim,
                        hidden_dim=args.hidden_dim, output_dim=output_dim,
                        feature_pre=args.feature_pre, layer_num=args.layer_num, dropout=args.dropout).to(device)
			# loss
			optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
			if 'link' in args.task:
				loss_func = nn.BCEWithLogitsLoss()
				out_act = nn.Sigmoid()

			batch_size_iter=512
			for epoch in range(args.epoch_num):
				if epoch==200:
					for param_group in optimizer.param_groups:
						param_group['lr'] /= 10
				model.train()
				optimizer.zero_grad()
				np.random.shuffle(data_list)
				effective_len = len(data_list)//args.batch_size*len(data_list)
				for id, data in enumerate(data_list[:effective_len]):
					model.to(device)
					if args.permute:
						preselect_anchor(data[k], layer_num=args.layer_num, anchor_num=args.anchor_num, device=device)
					for t in range(int(data[k].mask_link_positive_train.shape[1]/batch_size_iter)+1):
						start=t*batch_size_iter
						end=min(data[k].mask_link_positive_train.shape[1],(t+1)*batch_size_iter)
						out = model(data[k].x.to(device), data[k].edge_index[:,start:end].to(device))
						# get_link_mask(data,resplit=False)  # resample negative links
						edge_mask_train = np.concatenate((data[k].mask_link_positive_train[:,start:end], data[k].mask_link_negative_train[:,start:end]), axis=-1)
						nodes_first = torch.index_select(out, 0, torch.from_numpy(edge_mask_train[0,:]).long().to(device))
						nodes_second = torch.index_select(out, 0, torch.from_numpy(edge_mask_train[1,:]).long().to(device))
						pred = torch.sum(nodes_first * nodes_second, dim=-1)
						label_positive = torch.ones([data[k].mask_link_positive_train[:,start:end].shape[1],], dtype=pred.dtype)
						label_negative = torch.zeros([data[k].mask_link_negative_train[:,start:end].shape[1],], dtype=pred.dtype)
						label = torch.cat((label_positive,label_negative)).to(device)
						loss = loss_func(pred, label)

						# update
						loss.backward()
					if id % args.batch_size == args.batch_size-1:
						if args.batch_size>1:
							# if this is slow, no need to do this normalization
							for p in model.parameters():
								if p.grad is not None:
									p.grad /= args.batch_size
						optimizer.step()
						optimizer.zero_grad()


				if epoch % args.epoch_log == 0:
					# evaluate
					model.eval()
					loss_train = 0
					loss_val = 0
					loss_test = 0
					correct_train = 0
					all_train = 0
					correct_val = 0
					all_val = 0
					correct_test = 0
					all_test = 0
					auc_train = 0
					auc_val = 0
					auc_test = 0
					emb_norm_min = 0
					emb_norm_max = 0
					emb_norm_mean = 0
					prec_train = 0
					prec_val= 0
					prec_test =0
					recall_train=0
					recall_val=0
					recall_test=0
					f1_train=0
					f1_val=0
					f1_test=0
					prediction_train_array=[]
					label_train_array=[]
					prediction_valid_array=[]
					label_valid_array=[]
					prediction_test_array=[]
					label_test_array=[]
					for id, data in enumerate(data_list):
						model.cpu()
						out = model(data[k].x, data[k].edge_index)
						emb_norm_min += torch.norm(out.data, dim=1).min().cpu().numpy()
						emb_norm_max += torch.norm(out.data, dim=1).max().cpu().numpy()
						emb_norm_mean += torch.norm(out.data, dim=1).mean().cpu().numpy()

						# train
						# get_link_mask(data, resplit=False)  # resample negative links
						edge_mask_train = np.concatenate((data[k].mask_link_positive_train, data[k].mask_link_negative_train), axis=-1)
						nodes_first = torch.index_select(out, 0, torch.from_numpy(edge_mask_train[0, :]).long())
						nodes_second = torch.index_select(out, 0, torch.from_numpy(edge_mask_train[1, :]).long())
						pred = torch.sum(nodes_first * nodes_second, dim=-1)
						label_positive = torch.ones([data[k].mask_link_positive_train.shape[1], ], dtype=pred.dtype)
						label_negative = torch.zeros([data[k].mask_link_negative_train.shape[1], ], dtype=pred.dtype)
						label = torch.cat((label_positive, label_negative))
						loss_train += loss_func(pred, label).cpu().data.numpy()
						
						prediction=out_act(pred).flatten().data.cpu().numpy()
						auc_train += roc_auc_score(label.flatten().cpu().numpy(), prediction)
						prec_train += average_precision_score(label.flatten().cpu().numpy(), prediction)
						# print(prediction)
						prediction_train_array.append(prediction)
						label_train_array.append(label.flatten().cpu().numpy())
						
						ones=np.ones(prediction.shape)
						zeros=np.zeros(prediction.shape)
						class_pred=np.where(prediction>=0.5,ones,zeros)
						recall_train += recall_score(label.flatten().cpu().numpy(), class_pred)
						f1_train += f1_score(label.flatten().cpu().numpy(), class_pred)
						# print(label)
						# print(class_pred)

						# exit()
						# print(recall_val)
						# val
						edge_mask_val = np.concatenate((data[k].mask_link_positive_val, data[k].mask_link_negative_val), axis=-1)
						nodes_first = torch.index_select(out, 0, torch.from_numpy(edge_mask_val[0, :]).long())
						nodes_second = torch.index_select(out, 0, torch.from_numpy(edge_mask_val[1, :]).long())
						pred = torch.sum(nodes_first * nodes_second, dim=-1)
						label_positive = torch.ones([data[k].mask_link_positive_val.shape[1], ], dtype=pred.dtype)
						label_negative = torch.zeros([data[k].mask_link_negative_val.shape[1], ], dtype=pred.dtype)
						label = torch.cat((label_positive, label_negative))
						loss_val += loss_func(pred, label).cpu().data.numpy()
						auc_val += roc_auc_score(label.flatten().cpu().numpy(), out_act(pred).flatten().data.cpu().numpy())
						prec_val += average_precision_score(label.flatten().cpu().numpy(), out_act(pred).flatten().data.cpu().numpy())
						prediction=out_act(pred).flatten().data.cpu().numpy()
						ones=np.ones(prediction.shape)
						zeros=np.zeros(prediction.shape)
						class_pred=np.where(prediction>=0.5,ones,zeros)
						recall_val += recall_score(label.flatten().cpu().numpy(), class_pred)
						f1_val += f1_score(label.flatten().cpu().numpy(), class_pred)
						prediction_valid_array.append(prediction)
						label_valid_array.append(label.flatten().cpu().numpy())
						
						# print(recall_val)
						# test
						# print(data[k].mask_link_positive_test.shape[1])
						edge_mask_test = np.concatenate((data[k].mask_link_positive_test, data[k].mask_link_negative_test), axis=-1)
						nodes_first = torch.index_select(out, 0, torch.from_numpy(edge_mask_test[0, :]).long())
						nodes_second = torch.index_select(out, 0, torch.from_numpy(edge_mask_test[1, :]).long())
						pred = torch.sum(nodes_first * nodes_second, dim=-1)
						label_positive = torch.ones([data[k].mask_link_positive_test.shape[1], ], dtype=pred.dtype)
						label_negative = torch.zeros([data[k].mask_link_negative_test.shape[1], ], dtype=pred.dtype)
						label = torch.cat((label_positive, label_negative))
						loss_test += loss_func(pred, label).cpu().data.numpy()
						auc_test += roc_auc_score(label.flatten().cpu().numpy(), out_act(pred).flatten().data.cpu().numpy())
						prec_test += average_precision_score(label.flatten().cpu().numpy(), out_act(pred).flatten().data.cpu().numpy())
						prediction=out_act(pred).flatten().data.cpu().numpy()
						ones=np.ones(prediction.shape)
						zeros=np.zeros(prediction.shape)
						class_pred=np.where(prediction>=0.5,ones,zeros)
						recall_test += recall_score(label.flatten().cpu().numpy(), class_pred)
						f1_test += f1_score(label.flatten().cpu().numpy(), class_pred)
						prediction_test_array.append(prediction)
						label_test_array.append(label.flatten().cpu().numpy())
						# np.save("prediction_valid.txt", prediction)
						# np.save("label_valid.txt", label.flatten().cpu().numpy())

					loss_train /= id+1
					loss_val /= id+1
					loss_test /= id+1
					emb_norm_min /= id+1
					emb_norm_max /= id+1
					emb_norm_mean /= id+1
					auc_train /= id+1
					auc_val /= id+1
					auc_test /= id+1
					prec_train /= id+1
					prec_val /= id+1
					prec_test /= id+1
					recall_train /= id+1
					recall_val /= id+1
					recall_test /= id+1
					f1_train /= id+1
					f1_val /= id+1
					f1_test /= id+1
					np.save("prediction_train.npy", np.concatenate(prediction_train_array))
					np.save("label_train.npy", np.concatenate(label_train_array))
					np.save("prediction_valid.npy", np.concatenate(prediction_valid_array))
					np.save("label_valid.npy",  np.concatenate(label_valid_array))
					np.save("prediction_test.npy", np.concatenate(prediction_test_array))
					np.save("label_test.npy",  np.concatenate(label_test_array))

					print(repeat, epoch, 'Loss {:.4f}'.format(loss_train), 'Train AUC: {:.4f}'.format(auc_train),
						  'Val AUC: {:.4f}'.format(auc_val), 'Test AUC: {:.4f}'.format(auc_test), 'Train Prec: {:.4f}'.format(prec_train),
						  'Val Prec: {:.4f}'.format(prec_val), 'Test Prec: {:.4f}'.format(prec_test), 'Train Recall: {:.4f}'.format(recall_train),
						  'Val Recall: {:.4f}'.format(recall_val), 'Test Recall: {:.4f}'.format(recall_test), 'Train F1: {:.4f}'.format(f1_train),
						  'Val F1: {:.4f}'.format(f1_val), 'Test F1: {:.4f}'.format(f1_test))
					writer_train.add_scalar('repeat_' + str(repeat) + '/auc_'+dataset_name, auc_train, epoch)
					writer_train.add_scalar('repeat_' + str(repeat) + '/loss_'+dataset_name, loss_train, epoch)
					writer_val.add_scalar('repeat_' + str(repeat) + '/auc_'+dataset_name, auc_val, epoch)
					writer_train.add_scalar('repeat_' + str(repeat) + '/loss_'+dataset_name, loss_val, epoch)
					writer_test.add_scalar('repeat_' + str(repeat) + '/auc_'+dataset_name, auc_test, epoch)
					writer_test.add_scalar('repeat_' + str(repeat) + '/loss_'+dataset_name, loss_test, epoch)
					writer_test.add_scalar('repeat_' + str(repeat) + '/emb_min_'+dataset_name, emb_norm_min, epoch)
					writer_test.add_scalar('repeat_' + str(repeat) + '/emb_max_'+dataset_name, emb_norm_max, epoch)
					writer_test.add_scalar('repeat_' + str(repeat) + '/emb_mean_'+dataset_name, emb_norm_mean, epoch)
					result_val.append(auc_val)
					result_test.append(auc_test)


			result_val = np.array(result_val)
			result_test = np.array(result_test)
			results.append(result_test[np.argmax(result_val)])
		results = np.array(results)
		results_mean = np.mean(results).round(6)
		results_std = np.std(results).round(6)
		print('-----------------Final-------------------')
		print(results_mean, results_std)
		with open('results/{}_{}_{}_layer{}_approximate{}.txt'.format(args.task,args.model,dataset_name,args.layer_num,args.approximate), 'w') as f:
			f.write('{}, {}\n'.format(results_mean, results_std))

# export scalar data to JSON for external processing
writer_train.export_scalars_to_json("./all_scalars.json")
writer_train.close()
writer_val.export_scalars_to_json("./all_scalars.json")
writer_val.close()
writer_test.export_scalars_to_json("./all_scalars.json")
writer_test.close()
