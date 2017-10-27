#!/usr/bin/env python

import torch, cv2, sys
from darknet import Darknet19
import argparse, pdb, psycopg2, os, numpy as np, cStringIO, requests, json, re
from Dataset import RandomSampler
from torch.autograd import Variable
from utils.im_transform import imcv2_recolor
import utils.yolo as yolo_utils
import cfgs.config as cfg
from sklearn.cluster import DBSCAN
from shapely.geometry import box, MultiPolygon, mapping
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

def transform(im):
    im = imcv2_recolor(im)
    im = cv2.resize(im, (416, 416))
    return im

def process_file(filename, boxes, img_data, country, img_geom):
    boxes = np.concatenate(boxes, axis=0) if len(boxes) > 0 else boxes
    if len(boxes) > 0:
        labels = DBSCAN(eps=100, min_samples=1).fit(boxes[:, :2]).labels_
        if (labels >= 0).any():
            # Find the largest cluster
            values, counts = np.unique(labels[labels >= 0], return_counts=True)
            cluster_id = values[np.argmax(counts)]
            mp = MultiPolygon([box(*boxes[i]) for i, _ in enumerate(filter(lambda x: x == cluster_id, labels))])
            (cx,), (cy,) = mp.centroid.xy
            cx, cy = int(cx), int(cy)
        else:
            # No cluster was found, just center around the most confident box
            best_box = np.argmax(boxes[:, -1])
            cx = int(round(boxes[best_box, (0,2)].mean()))
            cy = int(round(boxes[best_box, (1,3)].mean()))

        xmin, xmax = cx - 250, cx + 250
        ymin, ymax = cy - 250, cy + 250

        if xmin < 0:
            dx = -xmin
            xmin, xmax = xmin + dx, xmax + dx
        if xmax > img_data.shape[1]:
            dx = xmax - img_data.shape[1]
            xmin, xmax = xmin - dx, xmax - dx
        if ymin < 0:
            dy = -ymin
            ymin, ymax = ymin + dy, ymax + dy
        if ymax > img_data.shape[0]:
            dy = ymax - img_data.shape[0]
            ymin, ymax = ymin - dy, ymax - dy

        boxes[:, (0, 2)] -= xmin
        boxes[:, (1, 3)] -= ymin

        boxes = np.clip(boxes, a_min=0, a_max=500)
        mask = (boxes[:,2] - boxes[:,0] >= 3) & (boxes[:,3] - boxes[:,1] >= 3)
        boxes = boxes[mask]

        features = [{'geometry' : mapping(box(*b)), 'type' : 'Feature', 'properties' : {}} for b in boxes]
        vdata = {'type' : 'FeatureCollection', 'features' : features}

        binary = cStringIO.StringIO()
        binary.write(cv2.imencode('.jpg', img_data[ymin:ymax, xmin:xmax, :])[1].tostring())
        binary.reset()

        data = {
            'vectordata' : json.dumps(vdata),
            'geom' : json.dumps(mapping(img_geom))
        }

        files = {
            'file' : binary
        }

        res = requests.post('https://aighmapper.ml/sample/%s' % country.replace('-overlap', ''), data=data, files=files)
        if res.status_code == 200:
            print('Successfully uploaded sample to server!')
            pdb.set_trace()
        else:
            print(res.text)
            return False
    return True

def sample(net, country, conn, max_samples = 20, batch_size = 16, thresh=0.5):
    gen = RandomSampler(conn, country, transform)
    upload_count = 0
    img_boxes = []
    cur_filename = None
    cur_whole_img = None
    cur_img_geom = None
    cur = conn.cursor()

    while upload_count < max_samples:
        inputs, originals, meta = zip(*[next(gen) for _ in range(batch_size)])
        inputs = Variable(torch.stack(inputs, 0).cuda(), volatile=True)

        bbox_pred, iou_pred, prob_pred = net(inputs)

        # to numpy
        bbox_pred = bbox_pred.data.cpu().numpy()
        iou_pred = iou_pred.data.cpu().numpy()
        prob_pred = prob_pred.data.cpu().numpy()

        for i in range(len(bbox_pred)):
            row_off, col_off, filename, whole_img, img_geom = meta[i]
            if filename != cur_filename:
                print('Finished processing file')
                res = process_file(cur_filename, img_boxes, cur_whole_img, country, cur_img_geom)
                if res and cur_filename:
                    cur.execute("UPDATE buildings.images SET done=true WHERE project=%s AND filename=%s", (country, cur_filename))
                    conn.commit()
                img_boxes = []
                cur_filename = filename
                cur_whole_img = whole_img
                cur_img_geom = img_geom

            bboxes, scores, cls_inds = yolo_utils.postprocess(
                np.expand_dims(bbox_pred[i], 0), 
                np.expand_dims(iou_pred[i], 0),
                np.expand_dims(prob_pred[i], 0),
                originals[i].shape,
                cfg, 
                thresh
            )
            if len(bboxes) > 3:

                img_data = originals[i].copy()
                for box in bboxes:
                    cv2.rectangle(img_data, tuple(box[:2]), tuple(box[2:]), (0,0,255))

                cv2.imwrite('test.jpg', img_data)

                bboxes[:, (0, 2)] += col_off
                bboxes[:, (1, 3)] += row_off
                img_boxes.append(np.concatenate([bboxes, np.expand_dims(scores, 1)], axis=1))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--country', help='Which country to process', required=True)
    parser.add_argument('--weights', required=True, help='Weight file to use')
    parser.add_argument('--num_samples', type=int, default=20, help='Number of samples to upload')
    parser.add_argument('--thresh', type=float, help='Confidence threshold')
    args = parser.parse_args()

    conn = psycopg2.connect(
        dbname='aigh',
        host=os.environ.get('DB_HOST', 'localhost'),
        user=os.environ.get('DB_USER', ''),
        password=os.environ.get('DB_PASSWORD', '')
    )

    checkpoint = torch.load(args.weights)

    net = Darknet19()
    net.load_state_dict(checkpoint['state_dict'])

    net.cuda()
    net.eval()

    if 'stats' in checkpoint:
        args.thresh = checkpoint['stats']['thresh']
    elif not args.thresh:
        print('Error: no threshold specified in checkpoint and missing command line arg')
        sys.exit(1)

    print('Using threshold: %f' % args.thresh)

    sample(net, args.country, conn, max_samples = args.num_samples, thresh=args.thresh)






