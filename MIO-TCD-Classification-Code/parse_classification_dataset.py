# -*- coding: utf-8 -*-
'''
Date : January 2017

Authors : Pierre-Marc Jodoin from the University of Sherbrooke

Description : code used to parse the MIO-TCD classification dataset,  classify
            each image and save results in the proper csv format.  Please see
            http://tcd.miovision.com/ for more details on the dataset

Execution : simply type the following command in a terminal:

   >> python parse_classification_dataset.py ./train/ your_results_train.csv
or
   >> python parse_classification_dataset.py ./test/ your_results_test.csv


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
from os import listdir
from os.path import isfile, join, splitext
from tqdm import tqdm
import numpy as np
import csv
import sys
import pickle
import cv2

classes = ['articulated_truck', 'bicycle', 'bus', 'car',
           'motorcycle', 'non-motorized_vehicle', 'pedestrian',
           'pickup_truck', 'single_unit_truck', 'work_van', 'background']

# Load SVM model
#svmClassifier = None

with open('./model.pk1', 'rb') as f:
    svmClassifier = pickle.load(f)

def get_hog_features(images, cell, block, nbins):
    """
    Returns HoG features for all images.
    
    Adapted from tutorial 4.
    """
    cell_size = (cell, cell)  # h x w in pixels
    block_size = (block, block)  # h x w in cells
    
    img = images[0]
    
    # create HoG Object
    # winSize is the size of the image cropped to an multiple of the cell size
    hog = cv2.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1],
                                  img.shape[0] // cell_size[0] * cell_size[0]),
                        _blockSize=(block_size[1] * cell_size[1],
                                    block_size[0] * cell_size[0]),
                        _blockStride=(cell_size[1], cell_size[0]),
                        _cellSize=(cell_size[1], cell_size[0]),
                        _nbins=nbins)
    
    n_cells = (img.shape[0] // cell_size[0], img.shape[1] // cell_size[1])
    
    hog_feats_list = []
    for img in images:        
        # Compute HoG features
        hog_feats = hog.compute(img)\
                       .reshape(n_cells[1] - block_size[1] + 1,
                                n_cells[0] - block_size[0] + 1,
                                block_size[0], block_size[1], nbins) \
                       .transpose((1, 0, 2, 3, 4))  # index blocks by rows first

        # hog_feats now contains the gradient amplitudes for each direction,for each cell of its group for each group.
        # Indexing is by rows then columns.
        
        # computation for BlockNorm
        gradients = np.full((n_cells[0], n_cells[1], 8), 0, dtype=float)
        cell_count = np.full((n_cells[0], n_cells[1], 1), 0, dtype=int)

        for off_y in range(block_size[0]):
            for off_x in range(block_size[1]):
                gradients[off_y:n_cells[0] - block_size[0] + off_y + 1,
                          off_x:n_cells[1] - block_size[1] + off_x + 1] += \
                    hog_feats[:, :, off_y, off_x, :]
                cell_count[off_y:n_cells[0] - block_size[0] + off_y + 1,
                           off_x:n_cells[1] - block_size[1] + off_x + 1] += 1

        # Average gradients
        gradients /= cell_count
        
        hog_feats_list.append(gradients.ravel())
        
    return hog_feats_list

def classify_image(path_to_image):
    '''
    Classify the image contained in 'path_to_image'.

    You may replace this line with a call to your classification method
    '''
    image = cv2.imread(path_to_image, cv2.IMREAD_GRAYSCALE)
    images = np.array([cv2.resize(image, (128, 128))])

    test_img_features = get_hog_features(images, 4, 4, 8)
    test_img_features = np.array([x.flatten() for x in test_img_features])
    # testing
    label = svmClassifier.predict(test_img_features)[0]

    return label


def parse_dataset(path_to_dataset,nb_images):
    '''
    Parse every image contained in 'path_to_dataset' (a path to the training
    or testing set), classify each image and save in a csv file the resulting
    assignment

    dataset_result: dict structure returned by the function.  It contains the
            label of each image
    '''
    llist = listdir(path_to_dataset)
    dataset_result = {}

    do_break = False

    for count, name in enumerate(tqdm(llist)):
        dn = join(path_to_dataset, name)
        if isfile(dn):
            label = classify_image(dn)
            file_nb, file_ext = splitext(name)
            dataset_result[file_nb] = label
            if count == nb_images-1 or do_break:
                do_break = True
                break

        else:
            file_list = listdir(dn)
            for count1, file_name in enumerate(reversed(file_list)):
                file_name_with_path = join(dn, file_name)
                label = classify_image(file_name_with_path)
                file_nb, file_ext = splitext(file_name)
                if file_nb in dataset_result.keys():
                    print('error! ', file_nb, dataset_result[file_nb], ' vs ', file_name_with_path)
                dataset_result[file_nb] = label
                if count1 == nb_images-1 or do_break:
                    do_break = True
                    break


    return dataset_result


def save_classification_result(dataset_result, output_csv_file_name):
    '''
    save the dataset_result (a dict structure containing the class of every image)
    into a valid csv file.
    '''

    csvfile = open(output_csv_file_name, 'w')
    fieldnames = ['file_name', 'class_label']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    for file_number in dataset_result.keys():
        writer.writerow({'file_name': str(file_number), 'class_label': dataset_result[file_number]})

    csvfile.close()

if __name__ == '__main__':

    if len(sys.argv) < 3:
        print("\nUsage : \n\t python parse_classification_dataset.py PATH OUTPUT_CSV_FILE_NAME NUMBER_OF_IMAGES\n")
        print("\t PATH : path to the training or the testing dataset")
        print("\t OUTPUT_CSV_FILE_NAME : name of the resulting csv file\n")
        print("\t NUMBER_OF_IMAGES : number of images to be testing on\n")
    else:
        print('\nProcessing: ', sys.argv[1], '\n')
        dataset_result = parse_dataset(sys.argv[1], int(sys.argv[3]))
        save_classification_result(dataset_result, sys.argv[2])
