#!/usr/bin/env python

import torch, cv2
import os, numpy as np, cPickle, psycopg2, argparse, pdb
from darknet import Darknet19
import utils.yolo as yolo_utils
import utils.network as net_utils
from utils.timer import Timer
from Dataset import InferenceGenerator
import cfgs.config as cfg
from dotenv import load_dotenv, find_dotenv
from torch.autograd import Variable
from utils.im_transform import imcv2_recolor
load_dotenv(find_dotenv())

def preprocess(fname):
    # return fname
    image = cv2.imread(fname)
    im_data = np.expand_dims(yolo_utils.preprocess_test(image, cfg.inp_size), 0)
    return image, im_data

def transform(im):
    im = imcv2_recolor(im)
    im = cv2.resize(im, (416, 416))
    return im

class GenLoader:
    def __init__(self, generator, batch_size=16):
        self.generator = generator
        self.batch_size = batch_size

    def __next__(self):
        batch = [next(self.generator) for _ in range(self.batch_size)]
        inputs, originals, meta = zip(*batch)
        return torch.stack(inputs, 0), originals, meta

    def next(self):
        return self.__next__()

    def __iter__(self):
        return self

def test_net(net, dataset, conn, max_per_image=300, thresh=0.5, vis=False):
    loader = GenLoader(dataset)
    _t = {'im_detect': Timer(), 'misc': Timer()}
    for inputs, originals, meta in loader:
        im_data = Variable(inputs.cuda(), volatile=True)

        _t['im_detect'].tic()
        bbox_pred, iou_pred, prob_pred = net(im_data)

        # to numpy
        bbox_pred = bbox_pred.data.cpu().numpy()
        iou_pred = iou_pred.data.cpu().numpy()
        prob_pred = prob_pred.data.cpu().numpy()

        for i in range(len(bbox_pred)):
            bboxes, scores, cls_inds = yolo_utils.postprocess(
                np.expand_dims(bbox_pred[i], 0), 
                np.expand_dims(iou_pred[i], 0),
                np.expand_dims(prob_pred[i], 0),
                originals[i].shape,
                cfg, 
                thresh
            )

            if len(bboxes) > 0:
                orig = originals[i].copy()

                for box in bboxes:
                    cv2.rectangle(orig, tuple(box[:2]), tuple(box[2:]), (0,0,255))

                cv2.imwrite('./test.jpg', orig)
                pdb.set_trace()


        # detect_time = _t['im_detect'].toc()

        # _t['misc'].tic()

        # pdb.set_trace()

        # for j in range(imdb.num_classes):
        #     inds = np.where(cls_inds == j)[0]
        #     if len(inds) == 0:
        #         all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
        #         continue
        #     c_bboxes = bboxes[inds]
        #     c_scores = scores[inds]
        #     c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(np.float32, copy=False)
        #     all_boxes[j][i] = c_dets

        # # Limit to max_per_image detections *over all classes*
        # if max_per_image > 0:
        #     image_scores = np.hstack([all_boxes[j][i][:, -1] for j in range(imdb.num_classes)])
        #     if len(image_scores) > max_per_image:
        #         image_thresh = np.sort(image_scores)[-max_per_image]
        #         for j in xrange(1, imdb.num_classes):
        #             keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
        #             all_boxes[j][i] = all_boxes[j][i][keep, :]
        # nms_time = _t['misc'].toc()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--country', required=True, help='Which country to process')
    parser.add_argument('--weights', required=True, help='Weight file to use')
    args = parser.parse_args()
    
    conn = psycopg2.connect(
        dbname='aigh',
        host=os.environ.get('DB_HOST', 'localhost'),
        user=os.environ.get('DB_USER', ''),
        password=os.environ.get('DB_PASSWORD', '')
    )
    dataset = InferenceGenerator(conn, args.country, transform=transform)

    net = Darknet19()
    net.load_state_dict(torch.load(args.weights)['state_dict'])

    net.cuda()
    net.eval()

    test_net(net, dataset, conn)

