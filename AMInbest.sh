export CUDA_VISIBLE_DEVICES=${X_SGE_CUDA_DEVICE}

exp_no=1
dataset=AMI
headnum=8
layernum=8
layer=l0
tag=_32_rnn_direct_${layer}

expdir=${dataset}_transformer/AMI_${layernum}_${headnum}${tag}

model=${expdir}/model.pt
echo ${model}

python rescore.py \
    --data data/AMI \
    --nbest rescore/time_sorted_eval.100bestlist \
    --model ${model} \
    --lm transformer_rnn_xl_${layer} \
    --lmscale 10 \
    --lookback 32 \
    --subbatchsize 100 \
    --cuda \
    --mem_len 32 \
    --logfile LOGs/nbestlog.txt \
    --ppl \
