import argparse
import os
import sys
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import dgl
import numpy as np
import torch
import time
import datetime

from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
from Nbody.nbody_dataloader import RIDataset

from Nbody.nbody_flags import get_flags

import models as t_pkg
torch.autograd.set_detect_anomaly(True)


def to_np(x):
    return x.cpu().detach().numpy()


def get_acc(pred, x_T, v_T, y=None, verbose=True):

    acc_dict = {}
    pred = to_np(pred)
    x_T = to_np(x_T)
    v_T = to_np(v_T)
    assert len(pred) == len(x_T)

    if verbose:
        y = np.asarray(y.cpu())
        _sq = (pred - y) ** 2
        acc_dict['mse'] = np.mean(_sq)

    _sq = (pred[:, 0, :] - x_T) ** 2
    acc_dict['pos_mse'] = np.mean(_sq)

    _sq = (pred[:, 1, :] - v_T) ** 2
    acc_dict['vel_mse'] = np.mean(_sq)

    return acc_dict


def train_epoch(epoch, model, loss_fnc, dataloader, optimizer, FLAGS):
    model.train()
    loss_epoch = 0

    num_iters = len(dataloader)
    for i, (g, y1, y2) in enumerate(dataloader):
        g = g.to(FLAGS.device)
        x_T = y1.to(FLAGS.device).view(-1, 3)
        v_T = y2.to(FLAGS.device).view(-1, 3)
        y = torch.stack([x_T, v_T], dim=1)

        optimizer.zero_grad()
        
        # run model forward and compute loss
        pred = model(g)
        loss = loss_fnc(pred, y)
        
        loss_epoch += to_np(loss)
        
        if torch.isnan(loss):
            import pdb
            pdb.set_trace()

        # backprop
        loss.backward()
        optimizer.step()

        # print to console
        if i % FLAGS.print_interval == 0:
            print(f"[{epoch}|{i}] loss: {loss:.5f}")

    # log train accuracy for entire epoch to wandb
    loss_epoch /= len(dataloader)
    return loss_epoch


def test_epoch(epoch, model, loss_fnc, dataloader, FLAGS, dT):
    model.eval()

    keys = ['pos_mse', 'vel_mse']
    acc_epoch = {k: 0.0 for k in keys}
    acc_epoch_blc = {k: 0.0 for k in keys}  # for constant baseline
    acc_epoch_bll = {k: 0.0 for k in keys}  # for linear baseline
    loss_epoch = 0.0
    total_counter = 0
    for i, (g, y1, y2) in enumerate(dataloader):
        num_batches = y1.shape[0]
        g = g.to(FLAGS.device)
        x_T = y1.view(-1, 3)
        v_T = y2.view(-1, 3)
        y = torch.stack([x_T, v_T], dim=1).to(FLAGS.device)
        
        pred = model(g).detach()
        
        loss_epoch += to_np(loss_fnc(pred, y)/len(dataloader))
        acc = get_acc(pred, x_T, v_T, y=y)
        for k in keys:
            acc_epoch[k] += acc[k] * num_batches
        total_counter += num_batches

        # eval constant baseline
        bl_pred = torch.zeros_like(pred)
        acc = get_acc(bl_pred, x_T, v_T, verbose=False)
        for k in keys:
            acc_epoch_blc[k] += acc[k]/len(dataloader)

        # eval linear baseline
        # Apply linear update to locations.
        bl_pred[:, 0, :] = dT * g.ndata['v'][:, 0, :]
        acc = get_acc(bl_pred, x_T, v_T, verbose=False)
        for k in keys:
            acc_epoch_bll[k] += acc[k] / len(dataloader)
    
    return {k: acc_epoch[k]/total_counter for k in keys}


class RandomRotation(object):
    def __init__(self):
        pass

    def __call__(self, x):
        M = np.random.randn(3, 3)
        Q, __ = np.linalg.qr(M)
        return x @ Q


def collate(samples):
    graphs, y1, y2 = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.stack(y1), torch.stack(y2)


def main(FLAGS, UNPARSED_ARGV):
    # Prepare data
    train_dataset = RIDataset(FLAGS, split='train')
    train_loader = DataLoader(train_dataset,
                              batch_size=FLAGS.batch_size,
                              shuffle=True,
                              collate_fn=collate,
                              num_workers=FLAGS.num_workers,
                              drop_last=True)

    test_dataset = RIDataset(FLAGS, split='test')
    test_loader = DataLoader(test_dataset,
                             batch_size=FLAGS.batch_size,
                             shuffle=False,
                             collate_fn=collate,
                             num_workers=FLAGS.num_workers,
                             drop_last=False) 

    # time steps
    assert train_dataset.data['delta_T'] == test_dataset.data['delta_T']
    assert train_dataset.data['sample_freq'] == test_dataset.data['sample_freq']

    dT = train_dataset.data['delta_T'] * train_dataset.data[
        'sample_freq'] * FLAGS.ri_delta_t

    FLAGS.train_size = len(train_dataset)
    FLAGS.test_size = len(test_dataset)
    assert len(test_dataset) < len(train_dataset)
    
    model = t_pkg.NBodyModel(num_layers=FLAGS.num_layers-1, num_hidden_channels=FLAGS.num_channels)
    model.to(FLAGS.device)

    # Optimizer settings
    optimizer = optim.Adam(model.parameters(), lr=FLAGS.lr)
    criterion = nn.MSELoss()
    criterion = criterion.to(FLAGS.device)
    task_loss = criterion

    for epoch in range(FLAGS.num_epochs):
        train_epoch(epoch, model, task_loss, train_loader, optimizer, FLAGS)
        test_acc = test_epoch(epoch, model, task_loss, test_loader, FLAGS, dT)
    
    print('test acc.: ', test_acc)
    
if __name__ == '__main__':
    FLAGS, UNPARSED_ARGV = get_flags()
    
    print("\n\nFLAGS:", FLAGS)
    print("UNPARSED_ARGV:", UNPARSED_ARGV, "\n\n")

    # Where the magic is
    try:
        main(FLAGS, UNPARSED_ARGV)
    except Exception:
        import pdb, traceback
        traceback.print_exc()
        pdb.post_mortem()
