#!/usr/bin/env python

import torch, cv2
from Dataset import RasterDataset as Dataset
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

def detection_collate(batch):
    inputs, boxes, labels, dontcares = zip(*batch)
    return torch.stack(inputs, 0), list(boxes), list(labels), list(dontcares)

net = Darknet19()
# net_utils.load_net(cfg.trained_model, net)
# pretrained_model = os.path.join(cfg.train_output_dir, 'darknet19_voc07trainval_exp1_63.h5')
# pretrained_model = cfg.trained_model
# net_utils.load_net(pretrained_model, net)
net.load_from_npz(cfg.pretrained_model, num_conv=18)
net.cuda()
net.train()
print('load net succ...')

# optimizer
start_epoch = 0
lr = cfg.init_learning_rate
optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)

use_tensorboard = False
remove_all_log = False
batch_size = 16
batch_per_epoch = 32
train_loss = 0
bbox_loss, iou_loss, cls_loss = 0., 0., 0.
cnt = 0
t = Timer()
step_cnt = 0

epochs = 100

def transform(im, boxes, labels):
    im = imcv2_recolor(im)
    im = cv2.resize(im, (416, 416))
    boxes = ((boxes / 300) * 416).round()
    return im, boxes, labels

dataset = Dataset('training_data', transform=transform)

dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=detection_collate)
batches_per_epoch = len(dataset) / batch_size

for epoch in range(epochs):
    for step, (inputs, gt_boxes, gt_classes, dontcare) in enumerate(dataloader):
        t.tic()

        inputs = Variable(inputs.cuda())

        # forward
        net(inputs, gt_boxes, gt_classes, dontcare)

        # backward
        loss = net.loss
        bbox_loss += net.bbox_loss.data.cpu().numpy()[0]
        iou_loss += net.iou_loss.data.cpu().numpy()[0]
        cls_loss += net.cls_loss.data.cpu().numpy()[0]
        train_loss += loss.data.cpu().numpy()[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cnt += 1
        step_cnt += 1
        duration = t.toc()
        if step % cfg.disp_interval == 0:
            train_loss /= cnt
            bbox_loss /= cnt
            iou_loss /= cnt
            cls_loss /= cnt
            print('epoch %d[%d/%d], loss: %.3f, bbox_loss: %.3f, iou_loss: %.3f, cls_loss: %.3f (%.2f s/batch)' % (
                epoch, step, batches_per_epoch, train_loss, bbox_loss, iou_loss, cls_loss, duration))

            train_loss = 0
            bbox_loss, iou_loss, cls_loss = 0., 0., 0.
            cnt = 0
            t.clear()

    if epoch in cfg.lr_decay_epochs:
        lr *= cfg.lr_decay
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)

    save_name = os.path.join(cfg.train_output_dir, '{}_{}.h5'.format(cfg.exp_name, epoch))
    net_utils.save_net(save_name, net)
    print('save model: {}'.format(save_name))
    step_cnt = 0




