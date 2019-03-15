# # Localization with SVM classifier
# Uses multiscale sliding window with SVM to detect DICE coefficient,
# accuracy, precision and recall on localization dataset.

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import time
import imutils
import csv
import pickle
from scipy.spatial import distance
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

# Load SVM model
with open('../MIO-TCD-Classification-Code/model.pk1', 'rb') as f:
    svmClassifier = pickle.load(f)

# Extract ground truth localization values from gt_train.csv 
ground_truth_id = []
ground_truth_class = []
ground_truth_coordinates = []

with open('../MIO-TCD-Localization/gt_train.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    lines = 0
    for row in csv_reader:
        ground_truth_id.append(row[0])
        ground_truth_class.append(row[1])
        coordinates = list(map(int, [row[2], row[3], row[4], row[5]]))
        ground_truth_coordinates.append(coordinates)
        lines += 1
            
ground_truth_id = np.array(ground_truth_id)
ground_truth_class = np.array(ground_truth_class)
ground_truth_coordinates = np.array(ground_truth_coordinates)

def getHogFeatures(images):
    cell_size = (4, 4)  # h x w in pixels
    block_size = (4, 4)  # h x w in cells
    nbins = 8  # number of orientation bins
    
    hog = cv2.HOGDescriptor(_winSize=(images[0].shape[1] // cell_size[1] * cell_size[1],
                                          images[0].shape[0] // cell_size[0] * cell_size[0]),
                                _blockSize=(block_size[1] * cell_size[1],
                                            block_size[0] * cell_size[0]),
                                _blockStride=(cell_size[1], cell_size[0]),
                                _cellSize=(cell_size[1], cell_size[0]),
                                _nbins=nbins)
    
    n_cells = (images[0].shape[0] // cell_size[0], images[0].shape[1] // cell_size[1])
    
    listOfHogFeatures = []
    for img in images:
        # create HoG Object
        # winSize is the size of the image cropped to an multiple of the cell size
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
        # turn gradient 32,32,8 3D array to 1D for training 
        reshaped_gradients = gradients.ravel()
        listOfHogFeatures.append(reshaped_gradients)
        
    return listOfHogFeatures
  
def calculate_dice_coefficient(box1, box2):
    """
    Calculates the dice coefficient given two bounding boxes
    """
    # XY coords of intersection rectangle
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
 
    # area of intersection rectangle
    intersection_rect = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
 
    # areas of prediction and ground-truth rectangles
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
 
    intersect_over_union = intersection_rect / float(box1_area + box2_area - intersection_rect)
    return (2 * intersect_over_union)/(intersect_over_union + 1)

def extract_localization_images(path, extracted_images, extracted_value, image_ids):
    """
    Extracts images from localization dataset and returns them with their numeric IDs
    """
    extracted_count = 0

    for file in os.listdir(path):
        if (extracted_count < extracted_value):
            img = cv2.cvtColor(cv2.imread(os.path.join(path + "/" + file)), cv2.COLOR_BGR2GRAY)
            extracted_images.append(img)
            extracted_count = extracted_count + 1 
            image_ids.append(""+file[:-4]) # remove '.jpg' extension
        else:
            break
                            
    return extracted_images

def sliding_window(image, stepSize, windowSize):
    """ 
    Performs a sliding window operation
    """
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def get_gt_values(img):
    """
    Returns the ground truth of an image given its ID.
    """
    classes = ground_truth_class[ground_truth_id == img]
    coords = ground_truth_coordinates[ground_truth_id == img]
    return classes, coords

def ms_sliding_window(localization_image, img, w, h, step_size, scales):
    """
    Sliding window in multiscale
    """
    patches = []
    patches_coordinates = []
    patches_classes = []
    
    for scale in scales:
        window_size = (int(w * scale), int(h * scale))
        color = tuple(np.random.randint(0, 255, 3).tolist())
        for (x, y, window) in sliding_window(localization_image, stepSize=step_size, windowSize=window_size):
            # ignore bad window size
            if window.shape[0] < int(w*scale) or window.shape[1] < int(h*scale): continue
            
            resized_window = cv2.resize(window,(w,h))
            window_hog_features = getHogFeatures([resized_window])
            prediction = svmClassifier.predict(window_hog_features)

            if (prediction != 1): # not a background
                patches.append(window)
                patches_coordinates.append([x,y,x+int(w*scale),y+int(w*scale)])
                patches_classes.append(prediction)
        
    return patches, patches_coordinates, np.array(patches_classes)

def calculate_localization_results(localization_image, gt_classes, gt_class_coordinates, patches, patches_coordinates, patches_classes):
    """
    Find best mean coefficient
    """
    dice_coefficient_values = []
    prediction_results = []
    for gt_class, gt_coordinates in zip(gt_classes, gt_class_coordinates):
        closest_coordinates = []
        for patch, p_coordinates in zip(patches, patches_coordinates):
            distance_from_gt = distance.euclidean(p_coordinates, gt_coordinates)
            closest_coordinates.append(distance_from_gt)

        closest_coordinates = np.array(closest_coordinates)
        min_index = np.argmin(closest_coordinates)
        min_coordinates = patches_coordinates[min_index]

        dice_coefficient = calculate_dice_coefficient(min_coordinates, gt_coordinates)
        dice_coefficient_values.append(dice_coefficient)

        prediction_results.append(patches_classes[min_index])
    
    mean_dice_coefficient = np.mean(dice_coefficient_values)
    return mean_dice_coefficient, prediction_results

def get_label(x):
    labels = np.array(['articulated_truck', 'background', 'bicycle', 'bus', 'car', 'motorcycle',
           'non-motorized_vehicle', 'pedestrian', 'pickup_truck', 'single_unit_truck', 'work_van'])
    return labels[x]

# Localization with HoG and SVM
# extract 500 localization images.

training_set_path= "../MIO-TCD-Localization/train"
train_images = []
train_value = 100
image_ids = []

extract_localization_images(training_set_path, train_images, train_value, image_ids)

# Cross Validation 
k_fold = KFold(n_splits=10)

accuracies = []
precisions = []
recalls = []
mean_dice_coeffs = []

train_images = np.array(train_images)
image_ids = np.array(image_ids)
validation_index = 0

for train_index, valid_index in k_fold.split(train_images):
    print("K-fold number: ", validation_index)
    valid_set = train_images[valid_index]
    valid_image_ids = image_ids[valid_index]
    
    dice_coefficient_images = []
    all_class_predictions = []
    all_ground_truth_classes = []

    w = 128
    h = 128
    step_size = 32 
    scales = [0.75, 1, 1.25, 1.5]
    
    # Start localization for validation set
    for localization_image, image_id in zip(valid_set,valid_image_ids):
        img = cv2.imread("../MIO-TCD-Localization/train/" + str(image_id) + ".jpg")
        
        patches, patches_coordinates, patches_classes = ms_sliding_window(localization_image, img, w, h, step_size, scales)

        gt_classes, gt_class_coordinates = get_gt_values(image_id)

        localization_image_dice, class_predictions = calculate_localization_results(img, gt_classes, gt_class_coordinates, patches, patches_coordinates, patches_classes)
        
        dice_coefficient_images.append(localization_image_dice)

        for prediction in class_predictions:
            all_class_predictions.append(get_label(prediction))
        for gt_class in gt_classes:
            all_ground_truth_classes.append(gt_class)
         
    mean_dice_coeffs.append(dice_coefficient_images)    
    
    accuracy = accuracy_score(all_class_predictions, all_ground_truth_classes)
    accuracies.append(accuracy)
    
    precision = precision_score(all_class_predictions, all_ground_truth_classes, average='micro')
    precisions.append(precision)
    
    recall = recall_score(all_class_predictions, all_ground_truth_classes, average='micro')
    recalls.append(recall)
    
    validation_index += 1

# distribution of validation set DICE Coefficient 
total_dice_coeff = []

image_index = []
for i in range(len(valid_set)):
    image_index.append(i)

for dice_coefficient in mean_dice_coeffs:    
    total_dice_coeff += dice_coefficient
    print("Avg DICE coefficient of set: ", str(np.mean(dice_coefficient)* 100) + "%")
    
image_index = []
for i in range(len(valid_set * 10)):
    image_index.append(i)
    
plt.figure(figsize=(10,5))
print("Mean DICE coefficient of all sets: ", str(np.mean(mean_dice_coeffs)* 100) + "%")
plt.plot(total_dice_coeff)
plt.show()

print("Avg SVM accuracy: ", str(np.mean(accuracies) * 100) + "%")
print("Standard SVM deviation : ", str(np.std(accuracies) * 100) + "%")
print("Avg SVM recall: ", str(np.mean(recalls) * 100) + "%")
print("Avg SVM precision: ", str(np.mean(precisions) * 100) + "%")

