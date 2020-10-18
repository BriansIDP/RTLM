export CUDA_VISIBLE_DEVICES=${X_SGE_CUDA_DEVICE}

exp_no=1
dataset=SWBD
headnum=8
layernum=24
layer=l0
tag=_64_direct_${layer}

expdir=${dataset}_transformer/${dataset}_${layernum}_${headnum}${tag}
lm=transformer_rnn_xl_${layer}

model=${expdir}/model.pt

python forwardSWBD.py \
    --data data/SWBD \
    --nbest rescore_rt/time_sorted_rt03.nbestlist \
    --model ${model} \
    --lm ${lm} \
    --lmscale 10 \
    --lookback 64 \
    --subbatchsize 100 \
    --cuda \
    --mem_len 64 \
    --logfile LOGs/nbestlog.txt \
    --ppl \
