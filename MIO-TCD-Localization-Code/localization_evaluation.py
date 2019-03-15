'''
Date : January 2017

Authors : Zhiming Luo from the University of Sherbrooke

Description : code used to compute localization metrics of the MIO-TCD
            localization dataset.  Please see http://tcd.miovision.com/
            for more details

Execution : simply type the following command in a terminal:

   >> python localization_evaluation.py gt_train.csv your_results_train.csv

NOTE: this code was developed and tested with Python 3.5.2 and Linux (Ubuntu 14.04)

Disclamer:

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

'''

from __future__ import print_function
import numpy as np
import csv
import sys


classes = ['articulated_truck', 'bicycle', 'bus', 'car', 'motorcycle',
           'motorized_vehicle', 'non-motorized_vehicle', 'pedestrian',
           'pickup_truck', 'single_unit_truck', 'work_van']

# Motorized-Vehicles are vehicles which are too small to be labeled into a specific category
# For evaluation, if a motorized_vehicle is detected into a following category,
# we consider it as a true positive detection.
motorized_vehicle_classes = ['articulated_truck', 'bus', 'car',
                             'pickup_truck', 'single_unit_truck', 'work_van']


def VOCap(rec, prec):
    '''
        Compute the average precision following the code in Pascal VOC toolkit
    '''
    mrec = np.array(rec).astype(np.float32)
    mprec = np.array(prec).astype(np.float32)
    mrec = np.insert(mrec, [0, mrec.shape[0]], [0.0, 1.0])
    mprec = np.insert(mprec, [0, mprec.shape[0]], [0.0, 0.0])

    for i in range(mprec.shape[0]-2, -1, -1):
        mprec[i] = max(mprec[i], mprec[i+1])

    i = np.ndarray.flatten(np.array(np.where(mrec[1:] != mrec[0:-1]))) + 1
    ap = np.sum(np.dot(mrec[i] - mrec[i-1], mprec[i]))
    return ap


def iou_ratio(bbox_1, bbox_2):
    '''
        Compute the IoU ratio between two bounding boxes
    '''

    bi = [max(bbox_1[0], bbox_2[0]), max(bbox_1[1], bbox_2[1]),
          min(bbox_1[2], bbox_2[2]), min(bbox_1[3], bbox_2[3])]

    iw = bi[2] - bi[0] + 1
    ih = bi[3] - bi[1] + 1

    ov = 0

    if iw > 0 and ih > 0:
        ua = (bbox_1[2] - bbox_1[0] + 1) * (bbox_1[3] - bbox_1[1] + 1) + \
             (bbox_2[2] - bbox_2[0] + 1) * (bbox_2[3] - bbox_2[1] + 1) - \
             iw * ih
        ov = iw * ih / float(ua)

    return ov


def compute_metric_class(gt, res, cls, minoverlap):
    '''
        Computes the localization metrics of a given class
            * precision-recall curve
            * average precision
    '''

    # loading the ground truth for class cls
    npos = 0
    gt_cls = {}
    for img in gt.keys():
        index = np.array(gt[img]['class']) == cls
        BB = np.array(gt[img]['bbox'])[index]
        det = np.zeros(np.sum(index[:]))
        npos += np.sum(index[:])
        gt_cls[img] = {'BB': BB,
                       'det': det}

    # loading the detection result
    score = np.array(res[cls]['score'])
    imgs = np.array(res[cls]['img'])
    BB = np.array(res[cls]['bbox'])

    # sort detections by decreasing confidence
    si = np.argsort(-score)
    imgs = imgs[si]
    BB = BB[si, :]

    # assign detections to ground truth objects
    nd = len(score)
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    for d in range(nd):
        img = imgs[d]

        bb = BB[d, :]
        ovmax = 0

        for j in range(len(gt_cls[img]['BB'])):
            bbgt = gt_cls[img]['BB'][j]
            ov = iou_ratio(bb, bbgt)
            if ov > ovmax:
                ovmax = ov
                jmax = j

        if ovmax >= minoverlap:
            if not gt_cls[img]['det'][jmax]:
                tp[d] = 1
                gt_cls[img]['det'][jmax] = 1
            else:
                fp[d] = 1
        else:
            fp[d] = 1

    # compute precision/recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp/npos
    prec = tp/(fp+tp)

    ap = VOCap(rec, prec)

    return rec, prec, ap


def csv_evaluation(gt_file, res_file, minoverlap):
    '''
        Computes the localization scores between gt_file and res_file.

        gt_file is the CSV file contains groundtruth and each line contains
        the information for an object please see "gt_train.csv" for an example.

            file_name, label and four numbers (x1, y1, x2, y2)
            (x1,y1) and (x2, y2) are the top-left and bottom-right of a bounding box
            (...)
            00000000,articulated_truck,43,25,109,55
            00000000,car,106,32,124,45
            00000001,bus,205,155,568,314
            00000001,bus,285,123,477,168
            (...)

       res_file is the CSV file contains the results from a given algorithm
       and each line contains:
           file_name, label, confidence score, x1, y1, x2, y2
           (...)
           00000015,car,1.0,187,45,242,77
           00000015,car,0.7,177,29,201,37
           00000016,car,0.6,260,252,424,379
           (...)
    '''

    gt = {}
    with open(gt_file, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        for row in reader:
            img = row[0]
            cls = row[1]
            bbox = np.array(row[2:]).astype('float32')

            if img in gt:
                gt[img]['class'].append(cls)
                gt[img]['bbox'].append(bbox)
            else:
                gt[img] = {}
                gt[img]['class'] = [cls]
                gt[img]['bbox'] = [bbox]
                gt[img]['motorized_vehicle'] = False

            # A flag indicates whether the image contains motorized vehicle or not
            if cls == 'motorized_vehicle':
                gt[img]['motorized_vehicle'] = True


    # Load the result CSV file and separate into different classes
    res = {}
    for cls in classes:
        res[cls] = {}
        res[cls]['img'] = []
        res[cls]['score'] = []
        res[cls]['bbox'] = []

    with open(res_file, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        for row in reader:
            img, cls, score = row[0], row[1], float(row[2])
            bbox = np.array(row[3:]).astype('float32')

            # Compute the iou with all ground truth bounding boxes
            # if the highest's class is 'motorized_vehicle'
            # then change the class to 'motorized_vehicle'
            if gt[img]['motorized_vehicle']:
                if cls in motorized_vehicle_classes:
                    ovmax = 0
                    label = cls
                    for k, gt_bb in enumerate(gt[img]['bbox']):
                        ov = iou_ratio(bbox, gt_bb)
                        if ov > ovmax:
                            label = gt[img]['class'][k]
                            ovmax = ov

                    if ovmax > minoverlap:
                        if label == 'motorized_vehicle':
                            cls = 'motorized_vehicle'

            res[cls]['img'].append(img)
            res[cls]['score'].append(score)
            res[cls]['bbox'].append(bbox)

    metrics = {}
    map = []
    for cls in classes:
        rec, prec, ap = compute_metric_class(gt, res, cls, minoverlap)
        metrics[cls] = {}
        metrics[cls]['recall'] = rec
        metrics[cls]['precision'] = prec
        metrics[cls]['ap'] = ap
        map.append(ap)

    metrics['map'] = np.mean(np.array(map))

    return metrics


if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("Usage : \n\t python localization_evaluation.py gt_train.csv your_results_train.csv")
    else:
        print('Computing score between ', sys.argv[1], ' and ', sys.argv[2], '\n')
        metrics = csv_evaluation(sys.argv[1], sys.argv[2], minoverlap = 0.5)

        for cls in classes:
            rec = metrics[cls]['recall']
            prec = metrics[cls]['precision']
            ap = metrics[cls]['ap']

            print('Average Precision of %s: %s\n' % (cls, ap))

        print('Mean Average Precision: %s\n' % metrics['map'])



