python -u main.py --task link --model GAT --dataset ppi --gpu --cuda 0 \
--repeat_num 5 --layer_num 3 --epoch_num 400 |& tee ppi_out_lp_5.txt