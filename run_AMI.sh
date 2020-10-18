export CUDA_VISIBLE_DEVICES=${X_SGE_CUDA_DEVICE}

#!/bin/bash
if [[ $1 == 'train' ]]; then
    echo 'Run training...'
    python train_rnn.py \
        --cuda \
        --data data/AMI/ \
        --dataset AMI \
        --work_dir AMI_transformer \
        --n_layer 8 \
        --d_model 512 \
        --div_val 1 \
        --n_head 8 \
        --d_head 64 \
        --d_inner 512 \
        --dropout 0.3 \
        --dropatt 0.3 \
        --optim adam \
        --warmup_step 5000 \
        --max_step 15000 \
        --lr 0.0002 \
        --batch_size 24 \
        --tgt_len 32 \
        --mem_len 32 \
        --ext_len 0 \
        --future_len 0 \
        --eval_tgt_len 32 \
        --eval-interval 1600 \
        --attn_type 0 \
        --scheduler cosine \
        --rnnenc \
        --rnndim 512 \
        --layerlist '0' \
        --pen_layerlist '0' \
        --merge_type direct \
        --p_scale 0.0000 \
        ${@:2}
elif [[ $1 == 'eval' ]]; then
    echo 'Run evaluation...'
    python eval.py \
        --cuda \
        --data data/AMI/ \
        --dataset AMI \
        --batch_size 1 \
        --tgt_len 32 \
        --ext_len 0 \
        --mem_len 0 \
        --future_len 32 \
        --split test \
        --work_dir AMI_transformer/20200922-112741 \
        ${@:2}
else
    echo 'unknown argment 1'
fi
