import sys
import os

this_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(this_path, os.pardir, os.pardir, os.pardir, os.pardir))
sys.path.append(root_path)

import os.path
import math
import json
from utilities import paths
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import nltk

import models.baseline.vqa.cyanogenoid.config as config
import models.baseline.vqa.cyanogenoid.data as data
import models.baseline.vqa.cyanogenoid.model as model
import models.baseline.vqa.cyanogenoid.utils as utils


def update_learning_rate(optimizer, iteration):
    lr = config.initial_lr * 0.5 ** (float(iteration) / config.lr_halflife)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


total_iterations = 0


def run(net, loader, optimizer, tracker, train=False, prefix='', epoch=0):
    """ Run an epoch over the given loader """
    if train:
        net.train()
        tracker_class, tracker_params = tracker.MovingMeanMonitor, {'momentum': 0.99}
    else:
        net.eval()
        tracker_class, tracker_params = tracker.MeanMonitor, {}
        answ = []
        idxs = []
        accs = []

    tq = tqdm(loader, desc='{} E{:03d}'.format(prefix, epoch), ncols=0)
    loss_tracker = tracker.track('{}_loss'.format(prefix), tracker_class(**tracker_params))
    acc_tracker = tracker.track('{}_acc'.format(prefix), tracker_class(**tracker_params))

    log_softmax = nn.LogSoftmax().cuda()
    for v, q, a, idx, q_len in tq:
        var_params = {
            'volatile': not train,
            'requires_grad': False,
        }
        v = Variable(v.cuda(async=True), **var_params)
        q = Variable(q.cuda(async=True), **var_params)
        a = Variable(a.cuda(async=True), **var_params)
        q_len = Variable(q_len.cuda(async=True), **var_params)

        out = net(v, q, q_len)
        nll = -log_softmax(out)
        loss = (nll * a / 10).sum(dim=1).mean()
        acc = utils.batch_accuracy(out.data, a.data).cpu()

        if train:
            global total_iterations
            update_learning_rate(optimizer, total_iterations)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_iterations += 1
        else:
            # store information about evaluation of this minibatch
            _, answer = out.data.cpu().max(dim=1)
            answ.append(answer.view(-1))
            accs.append(acc.view(-1))
            idxs.append(idx.view(-1).clone())

        loss_tracker.append(loss.item())
        # acc_tracker.append(acc.mean())
        for a in acc:
            acc_tracker.append(a.item())
        fmt = '{:.4f}'.format
        tq.set_postfix(loss=fmt(loss_tracker.mean.value), acc=fmt(acc_tracker.mean.value))

    if not train:
        answ = list(torch.cat(answ, dim=0))
        accs = list(torch.cat(accs, dim=0))
        idxs = list(torch.cat(idxs, dim=0))
        return answ, accs, idxs


def main():


    sd = torch.load('logs/2017-08-04_00.55.19.pth')

    target_name = os.path.join('logs', 'results.pth')
    print('will save to {}'.format(target_name))

    questions_path = paths.data_path('vqa/Questions/v2_OpenEnded_mscoco_val2014_questions.json')

    with open(questions_path, 'r') as fp:
        questions = json.load(fp)

    question_ids = list(map(lambda item: item['question_id'], questions['questions']))

    answers = {}
    vqa_ready = []

    ans_map = {v: k for k, v in sd['vocab']['answer'].items()}

    with open(paths.resources_path('predictions/beam_size_1/maxlen_20/gpt2.json'), 'r') as fp:
        preds = json.load(fp)
        good_indices = [int(k) for k in list(preds.keys())]

    print(len(good_indices))

    num_tokens = len(sd['vocab']['question']) + 1

    net = model.Net(num_tokens)

    sd_weight_keys = list(sd['weights'].keys())
    net_weight_keys = list(net.state_dict().keys())

    for bad, good in zip(sd_weight_keys, net_weight_keys):
        sd['weights'][good] = sd['weights'].pop(bad)

    net.load_state_dict(sd['weights'])
    net.to('cuda')

    optimizer = optim.Adam([p for p in net.parameters() if p.requires_grad])

    tracker = utils.Tracker()
    val_loader = data.get_loader(val=True)
    r = run(net, val_loader, optimizer, tracker, train=False, prefix='val', epoch=0)
    """

    dt = {
        'ans': []
        ,
        'ids': []

    }

    for ans, idx in tqdm(zip(r[0], r[2])):
        dt['ids'].append(idx.item())
        dt['ans'].append(ans.item())

    with open('tmp.json', 'w+') as fp:
        json.dump(dt, fp)



    sd = torch.load('logs/2017-08-04_00.55.19.pth')

    target_name = os.path.join('logs', 'results.pth')
    print('will save to {}'.format(target_name))

    with open(paths.resources_path('100K_predictions/beam_size_1/maxlen_20/gpt2.json'), 'r') as fp:
        preds = json.load(fp)
        good_indices = [int(k) for k in list(preds.keys())]

    print(len(good_indices))

    answers = {}
    vqa_ready = []

    ans_map = {v: k for k, v in sd['vocab']['answer'].items()}

    with open('tmp.json', 'r') as fp:
        r = json.load(fp)

    bad = 0
    kept = 0
    for answer, idx in tqdm(zip(r['ans'], r['ids'])):
        if idx in good_indices:
            answers[idx] = nltk.word_tokenize(ans_map[answer])
            vqa_ready.append({
                'question_id': idx,
                'answer': ans_map[answer]
            })
            kept += 1
        else:
            bad += 1

    print('Discarded {} elements, kept {} elements'.format(bad, kept))

    with open(paths.resources_path('100K_predictions/beam_size_1/maxlen_20/vqa_ready_vqa_baseline.json'), 'w+') as fp:
        json.dump(vqa_ready, fp)

    with open(paths.resources_path('100K_predictions/beam_size_1/maxlen_20/vqa_baseline.json'), 'w+') as fp:
        json.dump(answers, fp)
    """

if __name__ == '__main__':
    main()
