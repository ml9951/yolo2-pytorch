#!/usr/bin/env python

import torch, cv2, json, argparse
from Dataset import Dataset
import os, numpy as np, datetime
from darknet import Darknet19
import utils.network as net_utils
from utils.timer import Timer
import cfgs.config as cfg
import pdb
from torch.multiprocessing import Pool
from torch.utils import data
from torch.autograd import Variable
from utils.im_transform import imcv2_recolor

parser = argparse.ArgumentParser()
parser.add_argument('--resume', help='Checkpoint .pth file to resume execution at')
args = parser.parse_args()

def detection_collate(batch):
    inputs, boxes, labels, dontcares = zip(*batch)
    return torch.stack(inputs, 0), list(boxes), list(labels), list(dontcares)

net = Darknet19()

if args.resume:
    checkpoint = torch.load(args.resume)
    start_epoch = checkpoint['epoch'] + 1
    optimizer = checkpoint['optimizer']
    net.load_state_dict(checkpoint['state_dict'])
    print('Resuming from checkpoint at epoch %d' % start_epoch)
else:
    start_epoch = 0
    lr = cfg.init_learning_rate
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    net.load_from_npz(cfg.pretrained_model, num_conv=18)

net.cuda()
net.train()
print('load net succ...')

use_tensorboard = False
remove_all_log = False
batch_size = 16
batch_per_epoch = 32
train_loss = 0
bbox_loss, iou_loss, cls_loss, batch_loss = 0., 0., 0., 0.
cnt = 0
t = Timer()

epochs = 150

def checkpoint(net, optim, checkpoint_name, epoch):
    state_dict = net.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()

    torch.save({
        'epoch': epoch,
        'state_dict': state_dict,
        'optimizer': optim},
        checkpoint_name)

def transform(im, boxes, labels):
    im = imcv2_recolor(im)
    im = cv2.resize(im, (416, 416))
    boxes = ((boxes / 300) * 416).round()
    return im, boxes, labels

train_data = json.load(open('../data/train_data.json'))
val_data = json.load(open('../data/val_data.json'))

training_set = Dataset('../data', train_data, transform=transform).even()
validation_set = Dataset('../data', val_data, transform=transform)

training_loader = data.DataLoader(
    training_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    collate_fn=detection_collate,
    drop_last=True
)

validation_loader = data.DataLoader(
    validation_set,
    batch_size=batch_size,
    collate_fn=detection_collate,
    num_workers=2,
    drop_last=True
)

best_loss = float('inf')

for epoch in range(start_epoch, epochs):
    epoch_loss = 0
    net.train()
    # Train
    for step, (inputs, gt_boxes, gt_classes, dontcare) in enumerate(training_loader):
        t.tic()
        optimizer.zero_grad()

        inputs = Variable(inputs.cuda())

        # forward
        net(inputs, gt_boxes, gt_classes, dontcare)

        # backward
        loss = net.loss
        bbox_loss += net.bbox_loss.data.cpu().numpy()[0]
        iou_loss += net.iou_loss.data.cpu().numpy()[0]
        cls_loss += net.cls_loss.data.cpu().numpy()[0]
        train_loss += loss.data.cpu().numpy()[0]
        epoch_loss += loss.data[0]
        loss.backward()
        optimizer.step()
        duration = t.toc()
        if step % cfg.disp_interval == 0:
            print('epoch %d[%d/%d], loss: %.3f, bbox_loss: %.3f, iou_loss: %.3f, cls_loss: %.3f (%.2f s/batch)' % (
                epoch, step, len(training_set) / batch_size, train_loss, bbox_loss, iou_loss, cls_loss, duration))

            train_loss = 0
            bbox_loss, iou_loss, cls_loss = 0., 0., 0.
            t.clear()

    if epoch in cfg.lr_decay_epochs:
        lr *= cfg.lr_decay
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)

    test_loss = 0
    for step, (inputs, gt_boxes, gt_classes, dontcare) in enumerate(validation_loader):
        inputs = Variable(inputs.cuda(), volatile=True)

        # forward
        net(inputs, gt_boxes, gt_classes, dontcare)

        # backward
        loss = net.loss
        bbox_loss += net.bbox_loss.data[0]
        iou_loss += net.iou_loss.data[0]
        cls_loss += net.cls_loss.data[0]
        batch_loss += net.loss.data[0]
        test_loss += loss.data[0]
        if step % cfg.disp_interval == 0:
            print('epoch %d[%d/%d], loss: %.3f, bbox_loss: %.3f, iou_loss: %.3f, cls_loss: %.3f' % (
                epoch, step, len(validation_set) / batch_size, batch_loss, bbox_loss, iou_loss, cls_loss))
            bbox_loss, iou_loss, cls_loss, batch_loss = 0., 0., 0., 0.

    if test_loss < best_loss:
        best_loss = test_loss
        save_name = os.path.join(cfg.train_output_dir, '%s_best.pth' % cfg.exp_name)
        checkpoint(net, optimizer, save_name, epoch)
        print('save model: %s' % save_name)




