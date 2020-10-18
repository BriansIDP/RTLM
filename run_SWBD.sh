export CUDA_VISIBLE_DEVICES=${X_SGE_CUDA_DEVICE}

#!/bin/bash
if [[ $1 == 'train' ]]; then
    echo 'Run training...'
    python train_rnn.py \
        --cuda \
        --data data/SWBD/ \
        --work_dir SWBD_transformer \
        --dataset AMI \
        --n_layer 24 \
        --d_model 512 \
        --div_val 4 \
        --n_head 8 \
        --d_head 64 \
        --d_inner 1024 \
        --dropout 0.2 \
        --dropatt 0.1 \
        --optim adam \
        --warmup_step 2000 \
        --max_step 200000 \
        --lr 0.00025 \
        --batch_size 32 \
        --tgt_len 64 \
        --mem_len 64 \
        --ext_len 0 \
        --future_len 0 \
        --eval_tgt_len 64 \
        --eval-interval 1600 \
        --attn_type 0 \
        --scheduler cosine \
        --pre_lnorm \
        --rnnenc \
        --rnndim 512 \
        --layerlist '0' \
        --merge_type project \
        --log-interval 200 \
        ${@:2}
elif [[ $1 == 'eval' ]]; then
    echo 'Run evaluation...'
    python eval.py \
        --cuda \
        --data ../data/SWBD/ \
        --dataset AMI \
        --batch_size 1 \
        --tgt_len 128 \
        --mem_len 128 \
        --split test \
        --same_length \
        --work_dir SWBD_transformer/20200715-081509 \
        ${@:2}
else
    echo 'unknown argment 1'
fi
