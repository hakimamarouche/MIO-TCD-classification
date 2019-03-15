'''
Date : January 2017

Authors : Zhiming Luo, Pierre-Marc Jodoin from the University of Sherbrooke

Description : code used to visualize some images from the MIO-TCD localization
            training dataset. The bounding boxes are in the train.csv file
            provided with the dataset.  Please see http://tcd.miovision.com/
            for more details on the dataset

Execution : simply type the following command in a terminal:

   >> python view_bounding_boxes.py ./train/ gt_train.csv
or
   >> python view_bounding_boxes.py ./test/ your_results_test.csv

NOTE: this code was developed and tested with Python 3.5.2 and Linux
      (Ubuntu 14.04)

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

from seaborn import color_palette
import numpy as np
import cv2
import os
import csv
import sys

classes = ['articulated_truck', 'bicycle', 'bus', 'car',
           'motorcycle', 'motorized_vehicle', 'non-motorized_vehicle',
           'pedestrian', 'pickup_truck',
           'single_unit_truck', 'work_van']


def make_color_map():
    '''
        Create a color map for each class
    '''
    names = sorted(set(classes))
    n = len(names)

    if n == 0:
        return {}
    cp = color_palette("Paired", n)

    cp[:] = [tuple(int(255*c) for c in rgb) for rgb in cp]

    return dict(zip(names, cp))


def plot_bboxes(img, bboxes, color_map):
    '''
        Plot the  bounding boxes of a given image with a pre-defined colormap
    '''
    show_img = np.copy(img)

    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.4
    thickness = 1

    for bbox in bboxes:
        label = bbox['class']
        pts = bbox['bbox']

        pt1 = (int(pts[0]), int(pts[1]))
        pt2 = (int(pts[2]), int(pts[3]))

        cv2.rectangle(show_img, pt1, pt2, color_map[label], 2)

        textSize, baseline = cv2.getTextSize(label, fontFace=fontFace,
                                             fontScale=scale,
                                             thickness=thickness)

        cv2.rectangle(show_img, pt1, (pt1[0]+textSize[0], pt1[1]+textSize[1]),
                      color_map[label])

        cv2.putText(show_img, label, (pt1[0], pt1[1]+baseline*2),
                    fontFace, scale, (255, 255, 255), thickness)

    return show_img


def main():

    if len(sys.argv) < 3:
        print("\nUsage : \n\t python view_bounding_boxes.py PATH CSV_FILE_NAME\n")
        print("\t PATH : path to the folder containing the training images ")
        print("\t CSV_FILE_NAME : csv file containing the bounding boxes (typically gt_train.csv) \n")
        return

    # Convert the CSV format ground truth into a python dictionary
    train_label = {}
    with open(sys.argv[2], 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            img_name = row[0]

            bbox = {}
            bbox['class'] = row[1]
            if len(row) == 6:
                bbox['bbox'] = np.array(row[2:]).astype('int32')
            else:
                bbox['score'] = float(row[2])
                bbox['bbox'] = np.array(row[3:]).astype('int32')

            if img_name in train_label:
                train_label[img_name].append(bbox)
            else:
                train_label[img_name] = [bbox]

    # Create a color map for each class
    color_map = make_color_map()

    # Plot the bounding box of the first 1000 images
    images = sorted(train_label.keys())

    for i in range(1000):
        img_name = images[i]
        path = os.path.join(sys.argv[1], img_name+'.jpg')
        bboxes = train_label[img_name]
        img = cv2.imread(path)
        if img is not None:
            print(img_name)
            show_img = plot_bboxes(img, bboxes, color_map)
            cv2.imshow('show', show_img)
            cv2.waitKey()

if __name__ == '__main__':
    main()
