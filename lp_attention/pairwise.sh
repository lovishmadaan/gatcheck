dataset='grid'

mkdir -p outputs/${dataset}
mkdir -p results/${dataset}

for layer_num in 3
do
    for hidden_dim in 256
    do
        for approximate in -1
        do
            for attention_heads in 4
            do
                for out_attention_heads in 6
                do
                    for activation in 'relu'
                    do
                        a=outputs/${dataset}/${dataset}_l${layer_num}_h${hidden_dim}_a${approximate}_ina${attention_heads}_outa${out_attention_heads}_act${activation}.txt
                        echo "${a}"
                        time python -u main.py --task link --dataset ${dataset} --model GAT_R --activation ${activation} --gpu --cuda 4 \
                        --repeat_num 5 --epoch_num 400 --layer_num ${layer_num} --hidden_dim ${hidden_dim} \
                        --approximate ${approximate} --attention_heads ${attention_heads} --out_attention_heads ${out_attention_heads} \
                        |& tee outputs/${dataset}/${dataset}_l${layer_num}_h${hidden_dim}_a${approximate}_ina${attention_heads}_outa${out_attention_heads}_act${activation}.txt
                    done
                done
            done
        done
    done
done