# RTLM
## Introduction

This repository contains the code for the paper "Transformer Language Models with LSTM-based Cross-utterance Information Representation". The code is mainly adapted from the [Transformer XL PyTorch implementation](https://github.com/kimiyoung/transformer-xl.git). Single GPU version is implemented.

## Prerequisite
PyTorch 1.0.0

## Training
To train on AMI text data<br>
`bash run_AMI.sh train --work_dir PATH_TO_WORK_DIR`

To train on SWBD text data<br>
`bash run_SWBD.sh train --work_dir PATH_TO_WORK_DIR`

## N-best Rescoring
Rescoring AMI nbest list
`bash AMInbest.sh`
Rescoring SWB nbest list
`bash SWBDnbest.sh`
Rescoring RT03 nbest list
`bash RTnbest.sh`

Note that the path to the trained LM (--model) and to the nbest list (--nbest) should be modified to your own path.
