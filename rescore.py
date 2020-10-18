# coding: utf-8
import argparse
import sys, os
import torch
import math
import time
from operator import itemgetter
import numpy as np
import torch.nn.functional as F

# import data

parser = argparse.ArgumentParser(description='PyTorch Level-2 RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./data/AMI',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='model.pt',
                    help='location of the 1st level model')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--lookback', type=int, default=0,
                    help='Number of backward utterance embeddings to be incorporated')
parser.add_argument('--uttlookforward', type=int, default=1,
                    help='Number of forward utterance embeddings to be incorporated')
parser.add_argument('--excludeself', type=int, default=1,
                    help='current utterance embeddings to be incorporated')
parser.add_argument('--saveprefix', type=str, default='tensors/AMI',
                    help='Specify which data utterance embeddings saved')
parser.add_argument('--nbest', type=str, default='dev.nbest.info.txt',
                    help='Specify which nbest file to be used')
parser.add_argument('--lmscale', type=float, default=6,
                    help='how much importance to attach to rnn score')
parser.add_argument('--lm', type=str, default='original',
                    help='Specify which language model to be used: rnn, ngram or original')
parser.add_argument('--ngramlist', type=str, default='',
                    help='Specify which ngram stream file to be used')
parser.add_argument('--saveemb', action='store_true',
                    help='save utterance embeddings')
parser.add_argument('--save1best', action='store_true',
                    help='save 1best list')
parser.add_argument('--context', type=str, default='0',
                    help='Specify which utterance embeddings to be used')
parser.add_argument('--use_context', action='store_true',
                    help='use future context')
parser.add_argument('--contextfile', type=str, default='rescore/time_sorted_dev.nbestlist.context',
                    help='1best file for context')
parser.add_argument('--logfile', type=str, default='LOGs/log.txt',
                    help='the logfile for this script')
parser.add_argument('--interp', action='store_true',
                    help='Linear interpolation of LMs')
parser.add_argument('--factor', type=float, default=0.8,
                    help='ngram interpolation weight factor')
parser.add_argument('--gscale', type=float, default=12.0,
                    help='ngram grammar scaling factor')
parser.add_argument('--maxlen', type=int, default=0,
                    help='how many future words to look at')
parser.add_argument('--use_true', action='store_true',
                    help='Use true dev file for study')
parser.add_argument('--true_file', type=str, default='data/dev.txt',
                    help='Specify which true context file to be used')
parser.add_argument('--futurescale', type=float, default=1.0,
                    help='how much importance to attach to future word scores')
parser.add_argument('--map', type=str, default='rescore/eval.map',
                    help='mapping file for utterance names')
parser.add_argument('--subbatchsize', type=int, default=20,
                    help='Sub batch size for batched forwarding')
parser.add_argument('--mem_len', type=int, default=0,
                    help='Sub batch size for batched forwarding')
parser.add_argument('--ppl', action='store_true',
                    help='Calculate and report ppl')
parser.add_argument('--pplword', action='store_true',
                    help='Calculate and report average ppl for each word')
parser.add_argument('--extra-model', type=str, default='',
                    help='Extra LM to be interpolated')
parser.add_argument('--extra-modeltype', type=str, default='RNN',
                    help='The type of the extra LM to be interpolated')

args = parser.parse_args()

def logging(s, print_=True, log_=True):
    if print_:
        print(s)
    if log_:
        with open(args.logfile, 'a+') as f_log:
            f_log.write(s + '\n')

# Read in dictionary
logging("Reading dictionary...")
dictionary = {}
with open(os.path.join(args.data, 'dictionary.txt')) as vocabin:
    lines = vocabin.readlines()
    for line in lines:
        ind, word = line.strip().split(' ')
        if word not in dictionary:
            dictionary[word] = ind
        else:
            logging("Error! Repeated words in the dictionary!")

ntokens = len(dictionary)
eosidx = int(dictionary['<eos>'])

if args.pplword:
    wordppl = {}
    for word, ind in dictionary.items():
        wordppl[word] = [0.0, 0]

device = torch.device("cuda" if args.cuda else "cpu")
cpu = torch.device("cpu")

# Read in trained 1st level model
logging("Reading model...")
with open(args.model, 'rb') as f:
    model = torch.load(f)
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    if args.cuda:
        model.cuda()

logging("Reading extra model...")
extramodel = None
if args.extra_model != '':
    with open(args.extra_model, 'rb') as f:
        extramodel = torch.load(f)
        # after load the rnn params are not a continuous chunk of memory
        # this makes them a continuous chunk, and will speed up forward pass
        if args.cuda:
            extramodel.cuda()
criterion = torch.nn.CrossEntropyLoss(reduction='none')

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def forward_extra(extra_model, inputs, targets, hidden):
    prob_list = []
    hidden_list = []
    for i, input_data in enumerate(inputs):
        target = targets[i]
        output, new_hidden = extra_model(input_data.view(-1, 1).to(device), hidden)
        probs = criterion(output.squeeze(1), target.to(device))
        probs = torch.exp(-probs)
        prob_list.append(probs)
        hidden_list.append(new_hidden)
    return prob_list, hidden_list

# Batched forward lookback RNN
def forward_each_utt_batched_lookback_rnn(model,
                                          lines,
                                          utt_name,
                                          prev_utts,
                                          mems,
                                          ppl=False,
                                          hidden=None,
                                          extra_model=None,
                                          extra_hidden=None):
    # Process each line
    inputs = []
    targets = []
    ac_scores = []
    lm_scores = []
    maxlen = 0
    # new_mems = mems
    utterances = []
    utterances_ind = []
    extra_inputs = []
    extra_targets = []
    target_index_list = []

    for line in lines:
        linevec = line.strip().split()
        ac_score = float(linevec[0])
        utterance = linevec[4:-1]
        currentline = []
        for i, word in enumerate(utterance):
            if word in dictionary:
                currentline.append(int(dictionary[word]))
            else:
                currentline.append(int(dictionary['<UNK>']))
        utterances.append(utterance)
        utterances_ind.append(currentline)
        ac_scores.append(ac_score)
        if len(currentline) > maxlen:
            maxlen = len(currentline)
    mask = []
    ac_score_tensor = torch.tensor(ac_scores).to(device)
    # Pad inputs and targets, prev_append in [0, len(prev_utts)]
    prev_append = max(min(args.lookback - maxlen, len(prev_utts)), 1)
    for i, symbols in enumerate(utterances_ind):
        full_sequence = prev_utts[-prev_append:] + symbols + [eosidx] * (maxlen - len(symbols) + 1)
        inputs.append(full_sequence[:-1])
        targets.append(full_sequence[1:])
        # get interpolated model inputs and targets
        extra_inputs.append(torch.LongTensor([eosidx] + symbols))
        extra_targets.append(torch.LongTensor(symbols+[eosidx]))
        mask.append([0.0] * (prev_append - 1) + [1.0] * (len(symbols) + 1) + [0.0] * (maxlen - len(symbols)))
    # arrange inputs and targets into tensors
    input_tensor = torch.LongTensor(inputs).to(device).t().contiguous()
    target_tensor = torch.LongTensor(targets).to(device).t().contiguous()
    mask_tensor = torch.tensor(mask).to(device).t().contiguous()
    bsize = input_tensor.size(1)
    seq_len = input_tensor.size(0)

    # forward prop interpolate model
    if args.extra_model != '' and args.extra_modeltype == 'RNN':
        interp_prob_list, extra_hidden_list = forward_extra(extra_model, extra_inputs, extra_targets, extra_hidden)

    # Forward prop transformer
    logProblist = []
    mem_list = []
    ppl_list = []
    hidden_list = []
    # initialise RNN hidden state
    if hidden is None and getattr(model, "rnnenc", False):
        hidden = model.init_hidden(1)
        pos_hidden = [(hid[0][-1:], hid[1][-1:]) for hid in hidden]
    elif prev_append == 1 and getattr(model, "rnnenc", False):
        pos_hidden = [(hid[0][-1:], hid[1][-1:]) for hid in hidden]
    elif getattr(model, "rnnenc", False):
        pos_hidden = [(hid[0][-prev_append:-prev_append+1], hid[1][-prev_append:-prev_append+1]) for hid in hidden]
    # import pdb; pdb.set_trace()
    # transformer XL
    tiled_mems = tuple()
    if len(mems) > 0 and prev_append < len(prev_utts):
        # determine how much memory to keep: prev_append + tiled_mems[0].size(0) = mems[0].size(0)
        tiled_mems = [mem[-prev_append-args.mem_len+1:-prev_append+1].repeat(1, bsize, 1) for mem in mems]
    # Start forwarding
    for i in range(0, bsize, args.subbatchsize):
        # mems for transformer XL
        if len(tiled_mems) > 0:
            this_mem = [mem[:, i:i+args.subbatchsize, :].contiguous() for mem in tiled_mems]
        else:
            this_mem = tuple()

        bsz = min(args.subbatchsize, bsize - i)
        # expand rnn hidden state
        rnn_hidden = None
        if hidden is not None:
            rnn_hidden = [(hid[0].repeat(1, bsz, 1), hid[1].repeat(1, bsz, 1)) for hid in pos_hidden]

        ret = model(input_tensor[:, i:i+args.subbatchsize].contiguous(),
                    target_tensor[:, i:i+args.subbatchsize].contiguous(),
                    *this_mem, rnn_hidden=rnn_hidden, stepwise=True)
        loss, this_mem, penalty, rnn_hidden = ret[0], ret[1:-2], ret[-2], ret[-1]

        if args.mem_len > 0 and len(this_mem) > 0:
            mem_list.append(torch.stack(this_mem))
        loss = loss * mask_tensor[:, i:i+args.subbatchsize]
        logProblist.append(loss)
        hidden_list.append(rnn_hidden)
        if args.pplword:
            ppl_list.append(loss)
        # outputlist.append(output[:,-1,:])
    lmscores = torch.cat(logProblist, 1)
    if args.extra_model == '':
        lmscores = torch.sum(lmscores, dim=0)
    else:
        interpolated_score = []
        for i, probs in enumerate(interp_prob_list):
            tranformer_score = lmscores[:,i].tolist()
            tranformer_score = torch.tensor([np.exp(-score) for score in tranformer_score if score > 0]).to(device)
            assert len(probs) == len(tranformer_score)
            lmscore = -torch.log(args.factor * tranformer_score + (1 - args.factor) * probs)
            interpolated_score.append(torch.sum(lmscore))
        lmscores = torch.stack(interpolated_score)
    # lmscores = torch.sum(logProb.view(seq_len, bsize)*mask_tensor, 0)
    total_scores = - lmscores * args.lmscale + ac_score_tensor
    # Get output in some format
    outputlines = []
    for i, utt in enumerate(utterances):
        out = ' '.join([utt_name+'-'+str(i+1), '{:5.2f}'.format(lmscores[i])])
        outputlines.append(out+'\n')
    max_ind = torch.argmax(total_scores)
    # RNN hidden state selection
    if len(hidden_list) > 0:
        all_hidden = []
        for hid in zip(*hidden_list):
            hid_l = list(zip(*hid))
            all_hidden.append((torch.cat(hid_l[0], dim=1), torch.cat(hid_l[1], dim=1)))
        best_hid = [(hid[0][:, max_ind:max_ind+1, :], hid[1][:, max_ind:max_ind+1, :]) for hid in all_hidden]
    else:
        best_hid = None

    best_utt = utterances[max_ind]
    best_utt_len = len(utterances[max_ind])
    prev_utts += (utterances_ind[max_ind] + [eosidx])
    hidden = [(torch.cat([hidden[i][0][-args.lookback-2:], hid[0][:best_utt_len+1]], dim=0),
                 torch.cat([hidden[i][1][-args.lookback-2:], hid[1][:best_utt_len+1]], dim=0)) for i, hid in enumerate(best_hid)]
    if len(mem_list) > 0:
        mem_list = torch.cat(mem_list, dim=2)
        start_pos = max(mem_list.size(1) - maxlen - 1, 0)
        mem_list = mem_list[:, start_pos:start_pos+len(best_utt)+1, max_ind:max_ind+1, :]
        if len(mems) > 0:
            mem_list = [torch.cat([mems[i], mem_list[i]])[-(args.mem_len+args.lookback):] for i in range(mem_list.size(0))]
        else:
            mem_list = [mem_list[i] for i in range(mem_list.size(0))]
    # extra hidden states for interpolation
    extrahidden = extra_hidden_list[max_ind] if args.extra_model != '' else None
    # calculate perplexity
    best_ppl = lmscores[max_ind] if ppl else None
    # calculate per word perplexity
    if args.pplword:
        ppl_list = torch.cat(ppl_list, dim=1)[:, max_ind]
        for i, word in enumerate(best_utt+['<eos>']):
            if word in wordppl:
                wordppl[word][0] += ppl_list[i+prev_append-1]
                wordppl[word][1] += 1
            else:
                wordppl['OOV'][0] += ppl_list[i+prev_append-1]
                wordppl['OOV'][1] += 1
    return best_utt, outputlines, prev_utts[-args.lookback:], mem_list, best_ppl, hidden, extrahidden

# Batched forward lookback
def forward_each_utt_batched_lookback(model,
                                      lines,
                                      utt_name,
                                      prev_utts,
                                      mems,
                                      ppl=False,
                                      hidden=None,
                                      extra_model=None,
                                      extra_hidden=None):
    # Process each line
    inputs = []
    targets = []
    ac_scores = []
    lm_scores = []
    maxlen = 0
    new_mems = mems
    utterances = []
    utterances_ind = []
    target_index_list = []
    extra_inputs = []
    extra_targets = []

    for line in lines:
        linevec = line.strip().split()
        ac_score = float(linevec[0])
        utterance = linevec[4:-1]
        currentline = []
        for i, word in enumerate(utterance):
            if word in dictionary:
                currentline.append(int(dictionary[word]))
            else:
                currentline.append(int(dictionary['<UNK>']))
        utterances.append(utterance)
        utterances_ind.append(currentline)
        ac_scores.append(ac_score)
        if len(currentline) > maxlen:
            maxlen = len(currentline)
    mask = []
    ac_score_tensor = torch.tensor(ac_scores).to(device)
    # Pad inputs and targets, prev_append in [0, len(prev_utts)]
    prev_append = max(min(args.lookback - maxlen, len(prev_utts)), 1)
    for i, symbols in enumerate(utterances_ind):
        full_sequence = prev_utts[-prev_append:] + symbols + [eosidx] * (maxlen - len(symbols) + 1)
        inputs.append(full_sequence[:-1])
        targets.append(full_sequence[1:])
        # get interpolated model inputs and targets
        extra_inputs.append(torch.LongTensor([eosidx] + symbols))
        extra_targets.append(torch.LongTensor(symbols+[eosidx]))

        mask.append([0.0] * (prev_append - 1) + [1.0] * (len(symbols) + 1) + [0.0] * (maxlen - len(symbols)))
    # arrange inputs and targets into tensors
    input_tensor = torch.LongTensor(inputs).to(device).t().contiguous()
    target_tensor = torch.LongTensor(targets).to(device).t().contiguous()
    mask_tensor = torch.tensor(mask).to(device).t().contiguous()
    bsize = input_tensor.size(1)
    seq_len = input_tensor.size(0)

    # forward prop interpolate model
    if args.extra_model != '' and args.extra_modeltype == 'RNN':
        interp_prob_list, extra_hidden_list = forward_extra(extra_model, extra_inputs, extra_targets, extra_hidden)

    # Forward prop transformer
    logProblist = []
    mem_list = []
    ppl_list = []
    # initialise RNN hidden stte
    if hidden is None and getattr(model, "rnnenc", False):
        hidden = model.init_hidden(1)
    # transformer XL
    tiled_mems = tuple()
    if len(mems) > 0 and prev_append < len(prev_utts):
        # determine how much memory to keep: prev_append + tiled_mems[0].size(0) = mems[0].size(0)
        tiled_mems = [mem[-prev_append-args.mem_len+1:-prev_append+1].repeat(1, bsize, 1) for mem in mems]
    for i in range(0, bsize, args.subbatchsize):
        # mems for transformer XL
        if len(tiled_mems) > 0:
            this_mem = [mem[:, i:i+args.subbatchsize, :].contiguous() for mem in tiled_mems]
        else:
            this_mem = tuple()

        bsz = min(args.subbatchsize, bsize - i)
        # expand rnn hidden state
        rnn_hidden = None
        if hidden is not None:
            rnn_hidden = [(hid[0].repeat(1, bsz, 1), hid[1].repeat(1, bsz, 1)) for hid in hidden]

        ret = model(input_tensor[:, i:i+args.subbatchsize].contiguous(),
                    target_tensor[:, i:i+args.subbatchsize].contiguous(),
                    *this_mem, rnn_hidden=rnn_hidden)
        loss, this_mem, penalty, hidden = ret[0], ret[1:-2], ret[-2], ret[-1]
        if args.mem_len > 0 and len(this_mem) > 0:
            mem_list.append(torch.stack(this_mem))
        loss = loss * mask_tensor[:, i:i+args.subbatchsize]
        logProblist.append(loss)
        if args.pplword:
            ppl_list.append(loss)
        # outputlist.append(output[:,-1,:])
    lmscores = torch.cat(logProblist, 1)
    if args.extra_model == '':
        lmscores = torch.sum(lmscores, dim=0)
    else:
        interpolated_score = []
        for i, probs in enumerate(interp_prob_list):
            tranformer_score = lmscores[:,i].tolist()
            tranformer_score = torch.tensor([np.exp(-score) for score in tranformer_score if score > 0]).to(device)
            assert len(probs) == len(tranformer_score)
            lmscore = -torch.log(args.factor * tranformer_score + (1 - args.factor) * probs)
            interpolated_score.append(torch.sum(lmscore))
        lmscores = torch.stack(interpolated_score)
    # lmscores = torch.sum(logProb.view(seq_len, bsize)*mask_tensor, 0)
    total_scores = - lmscores * args.lmscale + ac_score_tensor
    # Get output in some format
    outputlines = []
    for i, utt in enumerate(utterances):
        out = ' '.join([utt_name+'-'+str(i+1), '{:5.2f}'.format(lmscores[i])])
        outputlines.append(out+'\n')
    max_ind = torch.argmax(total_scores)
    best_utt = utterances[max_ind]
    prev_utts += (utterances_ind[max_ind] + [eosidx])
    if len(mem_list) > 0:
        new_mems = torch.cat(mem_list, dim=2)
        start_pos = max(new_mems.size(1) - maxlen - 1, 0)
        new_mems = new_mems[:, start_pos:start_pos+len(best_utt)+1, max_ind:max_ind+1, :]
        if len(mems) > 0:
            new_mems = [torch.cat([mems[i], new_mems[i]])[-(args.mem_len+args.lookback):] for i in range(new_mems.size(0))]
        else:
            new_mems = [new_mems[i] for i in range(new_mems.size(0))]
    # extra hidden states for interpolation
    extrahidden = extra_hidden_list[max_ind] if args.extra_model != '' else None
    # calculate perplexity
    best_ppl = lmscores[max_ind] if ppl else None
    # calculate per word perplexity
    if args.pplword:
        ppl_list = torch.cat(ppl_list, dim=1)[:, max_ind]
        for i, word in enumerate(best_utt+['<eos>']):
            if word in wordppl:
                wordppl[word][0] += ppl_list[i+prev_append-1]
                wordppl[word][1] += 1
            else:
                wordppl['OOV'][0] += ppl_list[i+prev_append-1]
                wordppl['OOV'][1] += 1
    return best_utt, outputlines, prev_utts[-args.lookback:], new_mems, best_ppl, extrahidden

# Batched forward
def forward_each_utt_batched(model, lines, utt_name, prev_utts, mems, ppl=False, hidden=None):
    # Process each line
    inputs = []
    targets = []
    ac_scores = []
    lm_scores = []
    maxlen = 0
    new_mems = mems
    utterances = []
    utterances_ind = []

    for line in lines:
        linevec = line.strip().split()
        ac_score = float(linevec[0])
        utterance = linevec[4:-1]
        currentline = []
        for i, word in enumerate(utterance):
            if word in dictionary:
                currentline.append(int(dictionary[word]))
            else:
                currentline.append(int(dictionary['<UNK>']))
        currentline = [eosidx] + currentline
        currenttarget = currentline[1:]
        currenttarget.append(eosidx)
        inputs.append(currentline)
        targets.append(currenttarget)
        utterances.append(utterance)
        utterances_ind.append(currenttarget)
        ac_scores.append(ac_score)
        if len(currentline) > maxlen:
            maxlen = len(currentline)
    mask = []
    ac_score_tensor = torch.tensor(ac_scores).to(device)
    for i, symbols in enumerate(inputs):
        inputs[i] = symbols + [eosidx] * (maxlen - len(symbols))
        targets[i] = targets[i] + [eosidx] * (maxlen - len(symbols))
        mask.append([1.0] * len(symbols) + [0.0] * (maxlen - len(symbols)))

    input_tensor = torch.LongTensor(inputs).to(device).t().contiguous()
    target_tensor = torch.LongTensor(targets).to(device).t().contiguous()
    mask_tensor = torch.tensor(mask).to(device).t().contiguous()
    bsize = input_tensor.size(1)
    seq_len = input_tensor.size(0)

    # Forward prop transformer
    logProblist = []
    mem_list = []
    ppl_list = []
    hidden_list = []
    if hidden is None and getattr(model, "rnnenc", False):
        hidden = model.init_hidden(1)
    # transformer XL
    prev_mem_len = 0
    if len(mems) > 0:
        prev_mem_len = mems[0].size(0)
        tiled_mems = [mem.repeat(1, bsize, 1) for mem in mems]
    for i in range(0, bsize, args.subbatchsize):
        # mems for transformer XL
        if len(mems) > 0:
            this_mem = [mem[:, i:i+args.subbatchsize, :].contiguous() for mem in tiled_mems]
        else:
            this_mem = mems
        bsz = min(args.subbatchsize, bsize - i)
        # expand rnn hidden state
        rnn_hidden = None
        if hidden is not None:
            rnn_hidden = [(hid[0].repeat(1, bsz, 1), hid[1].repeat(1, bsz, 1)) for hid in hidden]
        # forward pass
        ret = model(input_tensor[:, i:i+args.subbatchsize].contiguous(),
                    target_tensor[:, i:i+args.subbatchsize].contiguous(),
                    *this_mem, rnn_hidden=rnn_hidden)
        loss, this_mems, rnn_hidden = ret[0], ret[1:-1], ret[-1]
        if args.mem_len > 0:
            mem_list.append(torch.stack(this_mems))
        if hidden is not None:
            hidden_list.append(rnn_hidden)
        loss = loss * mask_tensor[:, i:i+args.subbatchsize]
        logProblist.append(torch.sum(loss, dim=0))
        # outputlist.append(output[:,-1,:])
    lmscores = torch.cat(logProblist, 0)
    # lmscores = torch.sum(logProb.view(seq_len, bsize)*mask_tensor, 0)
    total_scores = - lmscores * args.lmscale + ac_score_tensor
    # Get output in some format
    outputlines = []
    for i, utt in enumerate(utterances):
        out = ' '.join([utt_name+'-'+str(i+1), '{:5.2f}'.format(lmscores[i])])
        outputlines.append(out+'\n')
    max_ind = torch.argmax(total_scores)
    # choose best rnn hidden state
    if len(hidden_list) > 0:
        all_hidden = []
        for hid in zip(*hidden_list):
            hid_l = list(zip(*hid))
            all_hidden.append((torch.cat(hid_l[0], dim=1), torch.cat(hid_l[1], dim=1)))
        best_hid = [(hid[0][:, max_ind:max_ind+1, :], hid[1][:, max_ind:max_ind+1, :]) for hid in all_hidden]
    else:
        best_hid = None

    best_utt = utterances[max_ind]
    prev_utts += utterances_ind[max_ind]
    if len(mem_list) > 0:
        new_mems = torch.cat(mem_list, dim=2)
        start_pos = max(new_mems.size(1) - seq_len, 0)
        new_mems = new_mems[:, start_pos:start_pos+len(best_utt)+1, max_ind:max_ind+1, :]
        if len(mems) > 0:
            new_mems = [torch.cat([mems[i], new_mems[i]])[-args.mem_len:] for i in range(new_mems.size(0))]
        else:
            new_mems = [new_mems[i] for i in range(new_mems.size(0))]
    # calculate perplexity
    best_ppl = lmscores[max_ind] if ppl else None
    return best_utt, outputlines, prev_utts, new_mems, best_ppl, best_hid

def forward_nbest_utterance(model, nbestfile, extramodel=None):
    """ The main body of the rescore function. """
    model.eval()
    # decide if we calculate the average of the loss
    if args.interp:
        forwardCrit = torch.nn.CrossEntropyLoss(reduction='none')
    else:
        forwardCrit = torch.nn.CrossEntropyLoss()

    extrahidden = None
    if args.extra_model != '' and extramodel is not None:
        extramodel.eval()
        extrahidden = extramodel.init_hidden(1)
    # initialising variables needed
    ngram_cursor = 0
    lmscored_lines = []
    best_utt_list = []
    emb_list = []
    utt_idx = 0
    prev_utts = [eosidx] # * args.lookback
    start = time.time()
    best_hid = None
    best_ppl = None
    mems = tuple()

    total_ppl = torch.zeros(1)
    total_len = 0

    # Ngram used for lattice rescoring
    with open(nbestfile) as filein:
        with torch.no_grad():
            for utterancefile in filein:
                # Iterating over the nbest list
                labname = utterancefile.strip().split('/')[-1]
                # Read in ngram LM files for interpolation
                if args.interp:
                    ngram_probfile_name = ngram_listfile.readline()
                    ngram_probfile = open(ngram_probfile_name.strip())
                    ngram_prob_lines = ngram_probfile.readlines()
                future_context = future_context_dict[utt_idx] if args.use_context else []

                # Start processing each nbestlist
                with open(utterancefile.strip()) as uttfile:
                    uttlines = uttfile.readlines()

                uttscore = []
                # Do re-ranking batch by batch
                if not args.interp:
                    if args.lookback > 0 and getattr(model, "rnnenc", False):
                        bestutt, to_write, prev_utts, mems, best_ppl, best_hid, extrahidden = forward_each_utt_batched_lookback_rnn(
                            model, uttlines, labname, prev_utts, mems, args.ppl, best_hid,
                            extra_model=extramodel, extra_hidden=extrahidden)
                    elif args.lookback > 0:
                        bestutt, to_write, prev_utts, mems, best_ppl, extrahidden = forward_each_utt_batched_lookback(
                            model, uttlines, labname, prev_utts, mems, args.ppl,
                            extra_model=extramodel, extra_hidden=extrahidden)
                    else:
                        bestutt, to_write, prev_utts, mems, best_ppl, best_hid = forward_each_utt_batched(
                            model, uttlines, labname, prev_utts, mems, args.ppl, best_hid)
                    lmscored_lines += to_write
                utt_idx += 1
                best_utt_list.append((labname, bestutt))
                if args.ppl and best_ppl is not None:
                    total_ppl += torch.sum(best_ppl)
                    total_len += len(bestutt) + 1
                # Log every completion of n utterances
                if utt_idx % 100 == 0:
                    logging("current ppl is {:5.2f}".format(torch.exp(total_ppl / total_len).item()))
                    logging('rescored {} utterances, time overlapped {:6.2f}'.format(str(utt_idx), time.time()-start))
    # Write out renewed lmscore file
    with open(nbestfile+'.renew.'+args.lm, 'w') as fout:
        fout.writelines(lmscored_lines)
    # Save 1-best for later use for the context
    if args.save1best:
        # Write out for second level forwarding
        with open(nbestfile+'.context', 'w') as fout:
            for i, eachutt in enumerate(best_utt_list):
                linetowrite = '<eos> ' + ' '.join(eachutt[1]) + ' <eos>\n'
                fout.write(linetowrite)
    if args.ppl:
        print(total_len)
        print(torch.exp(total_ppl / total_len))

logging('getting utterances')
forward_nbest_utterance(model, args.nbest, extramodel)
if args.pplword:
    with open("word_ppl_{}".format(args.lm), 'w') as fin:
        for word, group in wordppl.items():
            total_ppl, total_count = group
            if total_count > 0:
                fin.write('{}\t\t{}\t\t{:5.3f}\n'.format(word, total_count, float(total_ppl/total_count)))
