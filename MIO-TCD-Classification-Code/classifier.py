import cv2
import numpy as np
import random
import math
import glob
import os
import time
import pickle
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, recall_score, precision_score


def get_resized_images_and_labels(path_to_dataset, number_images, image_size):
    """
    Loads images from train folder.
    
    Adapted from the provided code (MIO-TCD).
    """
    llist = os.listdir(path_to_dataset)
    images = []
    labels = []

    for name in tqdm(llist):
        dn = os.path.join(path_to_dataset, name)
        if os.path.isdir(dn):

            file_list = os.listdir(dn)
            for count, file_name in enumerate(file_list):
                fn = os.path.join(dn, file_name)
                image = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, (image_size, image_size))
                images.append(image)

                labels.append(name)
                
                if count == number_images-1:
                    break
    
    return np.asarray(images), np.asarray(labels)
  
    
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


def main():
    images, labels = get_resized_images_and_labels("../MIO-TCD-Classification/train", 200, 128)

    training_img_features = get_hog_features(images, 4, 4, 8)
    training_img_features = np.array([x.flatten() for x in training_img_features])

    model = svm.SVC(gamma=0.001, kernel='linear')
    model.fit(training_img_features, labels)
    
    # save model to file
    with open('./model.pk1', 'wb') as f:
        pickle.dump(model, f)


    # cross-validation
    scores = cross_val_score(model, training_img_features, labels, cv=10)
    print("Accuracy & STD: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    recall = cross_val_score(model, training_img_features, labels, cv=10, scoring='recall_micro')
    print("Recall: %0.2f", np.mean(recall))
    precision = cross_val_score(model, training_img_features, labels, cv=10, scoring='precision_micro')
    print("Precision: %0.2f", np.mean(precision))

if __name__ == "__main__":
    main()
