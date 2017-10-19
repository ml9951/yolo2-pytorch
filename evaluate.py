#!/usr/bin/env python

import os, pdb, argparse, json, torch, cv2, numpy as np, rtree, pandas
from darknet import Darknet19
from utils.im_transform import imcv2_recolor
from torch.autograd import Variable
import utils.yolo as yolo_utils
import cfgs.config as cfg
from tqdm import tqdm
from shapely.geometry import MultiPolygon, box

def transform(im):
    im = imcv2_recolor(im)
    return cv2.resize(im, (416, 416)).transpose((2,0,1))[(2,1,0),:,:]
    
def get_metrics(gt_boxes, pred_boxes):
    false_positives = 0
    true_positives = 0
    false_negatives = 0
    total_overlap = 0.0

    # Create the RTree out of the ground truth boxes
    idx = rtree.index.Index()
    for i, rect in enumerate(gt_boxes):
        idx.insert(i, tuple(rect))

    gt_mp = MultiPolygon([box(*b) for b in gt_boxes])
    pred_mp = MultiPolygon([box(*b) for b in pred_boxes])

    for rect in pred_boxes:
        best_jaccard = 0.0
        best_idx = None
        best_overlap = 0.0
        for gt_idx in idx.intersection(rect):
            gt = gt_boxes[gt_idx]
            intersection = (min(rect[2], gt[2]) - max(rect[0], gt[0])) * (min(rect[3], gt[3]) - max(rect[1], gt[1]))
            rect_area = (rect[2] - rect[0]) * (rect[3] - rect[1])
            gt_area = (gt[2] - gt[0]) * (gt[3] - gt[1])
            union = rect_area + gt_area - intersection
            jaccard = float(intersection) / float(union)
            if jaccard > best_jaccard:
                best_jaccard = jaccard
                best_idx = gt_idx
            if intersection > best_overlap:
                best_overlap = intersection
        if best_idx is None or best_jaccard < 0.25:
            false_positives += 1
        else:
            true_positives += 1
        total_overlap = best_overlap
    total_jaccard = total_overlap / (gt_mp.area + pred_mp.area - total_overlap) if len(gt_boxes) > 0 else None
    false_negatives = idx.count((0,0,500,500))
    return false_positives, false_negatives, true_positives, total_jaccard

def img_iter(test_set, data_dir):
    for sample in test_set:
        img = cv2.imread(os.path.join(data_dir, '..', sample['image_path']))

        boxes = np.array([[r['x1'], r['y1'], r['x2'], r['y2']] for r in sample['rects']])

        for i in range(0, 201, 200):
            for j in range(0, 201, 200):
                subset = img[i:i+300, j:j+300, :]
                current_boxes = boxes.copy()
                if len(current_boxes) > 0:
                    current_boxes[:, (0, 2)] -= j
                    current_boxes[:, (1, 3)] -= i
                    current_boxes = np.clip(current_boxes, a_min = 0, a_max = 300)
                    mask = (current_boxes[:, 2] - current_boxes[:, 0] > 2) & (current_boxes[:, 3] - current_boxes[:, 1] > 2)
                    current_boxes = current_boxes[mask, :]

                yield torch.from_numpy(transform(subset.copy())).float(), subset, current_boxes

def test_net(net, test_set, data_dir, batch_size = 16, thresh=0.5):
    dataset = img_iter(test_set, data_dir)
    results = []
    for i in tqdm(range(0, len(test_set), batch_size)):
        inputs, originals, targets = zip(*[next(dataset) for _ in range(batch_size)])
        inputs = Variable(torch.stack(inputs, 0).cuda(), volatile=True)

        bbox_pred, iou_pred, prob_pred = net(inputs)

        # to numpy
        bbox_pred = bbox_pred.data.cpu().numpy()
        iou_pred = iou_pred.data.cpu().numpy()
        prob_pred = prob_pred.data.cpu().numpy()

        for i in range(len(bbox_pred)):
            bboxes, scores, cls_inds = yolo_utils.postprocess(
                np.expand_dims(bbox_pred[i], 0), 
                np.expand_dims(iou_pred[i], 0),
                np.expand_dims(prob_pred[i], 0),
                [300,300,3],
                cfg, 
                thresh
            )
            fp, fn, tp, jaccard = get_metrics(targets[i], bboxes)
            results.append({
                'false_positives' : fp,
                'false_negatives' : fn,
                'true_positives' : tp,
                'jaccard' : jaccard
            })

    pandas.DataFrame(results).to_csv('results.csv', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True, help='Weight file to use')
    parser.add_argument('--test_set', default='../data/val_data.json', help='JSON file describing the test data')
    args = parser.parse_args()
    
    data_dir = os.path.dirname(os.path.abspath(args.test_set))
    test_set = json.load(open(args.test_set))

    net = Darknet19()
    net.load_state_dict(torch.load(args.weights)['state_dict'])

    net.cuda()
    net.eval()

    test_net(net, test_set, data_dir)