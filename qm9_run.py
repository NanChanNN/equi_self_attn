import argparse
import os
import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import dgl
import math
import numpy as np
import torch

from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from qm9.QM9 import QM9Dataset

import models as t_pkg
torch.autograd.set_detect_anomaly(True)


def to_np(x):
    return x.cpu().detach().numpy()


def train_epoch(epoch, model, loss_fnc, dataloader, optimizer, scheduler, FLAGS):
    model.train()

    num_iters = len(dataloader)
    for i, (g, y) in enumerate(dataloader):
        g = g.to(FLAGS.device)
        y = y.to(FLAGS.device)

        optimizer.zero_grad()
        
        # run model forward and compute loss
        pred = model(g)
        l1_loss, __, rescale_loss = loss_fnc(pred, y)

        # backprop
        l1_loss.backward()
        optimizer.step()

        if i % FLAGS.print_interval == 0:
            print(f"[{epoch}|{i}] l1 loss: {l1_loss:.5f} rescale loss: {rescale_loss:.5f} [units]")
    
        scheduler.step(epoch + i / num_iters)


def val_epoch(epoch, model, loss_fnc, dataloader, FLAGS):
    model.eval()

    rloss = 0
    for i, (g, y) in enumerate(dataloader):
        g = g.to(FLAGS.device)
        y = y.to(FLAGS.device)

        # run model forward and compute loss
        pred = model(g).detach()
        __, __, rl = loss_fnc(pred, y, use_mean=False)
        rloss += rl
    rloss /= FLAGS.val_size

    print(f"...[{epoch}|val] rescale loss: {rloss:.5f} [units]")
    return to_np(rloss)


def test_epoch(epoch, model, loss_fnc, dataloader, FLAGS):
    model.eval()

    rloss = 0
    for i, (g, y) in enumerate(dataloader):
        g = g.to(FLAGS.device)
        y = y.to(FLAGS.device)

        # run model forward and compute loss
        pred = model(g).detach()
        __, __, rl = loss_fnc(pred, y, use_mean=False)
        rloss += rl
    rloss /= FLAGS.test_size

    print(f"...[{epoch}|test] rescale loss: {rloss:.5f} [units]")
    return to_np(rloss)


class RandomRotation(object):
    def __init__(self):
        pass

    def __call__(self, x):
        M = np.random.randn(3,3)
        Q, __ = np.linalg.qr(M)
        return x @ Q

    
def collate(samples):
    graphs, y = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(np.array(y))


def main(FLAGS, UNPARSED_ARGV):
    # Prepare data
    train_dataset = QM9Dataset(FLAGS.data_address, 
                               FLAGS.task,
                               mode='train', 
                               transform=RandomRotation(),
                               fully_connected=FLAGS.fully_connected)
    train_loader = DataLoader(train_dataset, 
                              batch_size=FLAGS.batch_size, 
			      shuffle=True, 
                              collate_fn=collate, 
                              num_workers=FLAGS.num_workers)

    val_dataset = QM9Dataset(FLAGS.data_address, 
                             FLAGS.task,
                             mode='valid', 
                             fully_connected=FLAGS.fully_connected) 
    val_loader = DataLoader(val_dataset, 
                            batch_size=FLAGS.batch_size, 
			    shuffle=False, 
                            collate_fn=collate, 
                            num_workers=FLAGS.num_workers)

    test_dataset = QM9Dataset(FLAGS.data_address, 
                             FLAGS.task, 
                             mode='test', 
                             fully_connected=FLAGS.fully_connected) 
    test_loader = DataLoader(test_dataset, 
                             batch_size=FLAGS.batch_size, 
			     shuffle=False, 
                             collate_fn=collate, 
                             num_workers=FLAGS.num_workers)

    FLAGS.train_size = len(train_dataset)
    FLAGS.val_size = len(val_dataset)
    FLAGS.test_size = len(test_dataset)

    # Construct the model
    if FLAGS.model == 'MyModel_OD':
        model = t_pkg.QM9Model(num_layers=FLAGS.num_layers, invariant_mod='OD', cross_product=False,
                               pooling=FLAGS.pooling, heads=FLAGS.head, div=FLAGS.div, 
                               hidden_dim=FLAGS.num_channels).to(FLAGS.device)
    elif FLAGS.model == 'MyModel_SOD':
        model = t_pkg.QM9Model(num_layers=FLAGS.num_layers, invariant_mod='SOD', cross_product=True,
                               pooling=FLAGS.pooling, heads=FLAGS.head, div=FLAGS.div, 
                               hidden_dim=FLAGS.num_channels).to(FLAGS.device)
    model.to(FLAGS.device)

    # Optimizer settings
    optimizer = optim.Adam(model.parameters(), lr=FLAGS.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 
                                                               FLAGS.num_epochs, 
                                                               eta_min=1e-4)
    
    # Loss function
    def task_loss(pred, target, use_mean=True):
        l1_loss = torch.sum(torch.abs(pred - target))
        l2_loss = torch.sum((pred - target)**2)
        if use_mean:
            l1_loss /= pred.shape[0]
            l2_loss /= pred.shape[0]
        
        rescale_loss = train_dataset.norm2units(l1_loss, FLAGS.task)
        return l1_loss, l2_loss, rescale_loss

    # Run training
    print('Begin training')
    for epoch in range(FLAGS.num_epochs):
        train_epoch(epoch, model, task_loss, train_loader, optimizer, scheduler, FLAGS)
        val_loss = val_epoch(epoch, model, task_loss, val_loader, FLAGS)
        test_loss = test_epoch(epoch, model, task_loss, test_loader, FLAGS)
    
    print('Task: ', FLAGS.task)
    print('val loss: ', val_loss, '\t test loss', test_loss)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model parameters
    parser.add_argument('--model', type=str, default='MyModel_SOD',
                        help="Model type")
    parser.add_argument('--num_layers', type=int, default=7,
            help="Number of equivariant layers")
    parser.add_argument('--num_channels', type=int, default=128,
            help="Number of channels in middle layers")
    parser.add_argument('--fully_connected', action='store_false',
            help="Include global node in graph")
    parser.add_argument('--div', type=int, default=2,
            help="Low dimensional embedding fraction")
    parser.add_argument('--pooling', type=str, default='sum',
            help="Choose from avg or max")
    parser.add_argument('--head', type=int, default=8,
            help="Number of attention heads")

    # Meta-parameters
    parser.add_argument('--batch_size', type=int, default=32, 
            help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-3, 
            help="Learning rate")
    parser.add_argument('--num_epochs', type=int, default=50, 
            help="Number of epochs")

    # Data
    parser.add_argument('--data_address', type=str, default='qm9/QM9_data.pt',
            help="Address to structure file")
    parser.add_argument('--task', type=str, default='homo',
            help="QM9 task ['homo, 'mu', 'alpha', 'lumo', 'gap', 'r2', 'zpve', 'u0', 'u298', 'h298', 'g298', 'cv']")

    # Logging
    parser.add_argument('--log_interval', type=int, default=25,
            help="Number of steps between logging key stats")
    parser.add_argument('--print_interval', type=int, default=250,
            help="Number of steps between printing key stats")

    # Miscellanea
    parser.add_argument('--num_workers', type=int, default=4, 
            help="Number of data loader workers")

    # Random seed for both Numpy and Pytorch
    parser.add_argument('--seed', type=int, default=2022)

    FLAGS, UNPARSED_ARGV = parser.parse_known_args()

    # Fix seed for random numbers
    torch.manual_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    # Automatically choose GPU if available
    if torch.cuda.is_available():
        FLAGS.device = torch.device('cuda:0')
    else:
        torch.device('cpu')

    print("\n\nFLAGS:", FLAGS)
    print("UNPARSED_ARGV:", UNPARSED_ARGV, "\n\n")

    # Where the magic is
    main(FLAGS, UNPARSED_ARGV)
