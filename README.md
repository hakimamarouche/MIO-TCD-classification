# MIO-TCD-classification
The following repository contains a computer vision approach for traffic localization and classification.

### Details of the MIO-TCD dataset
The dataset consists of total 786,702 images with 648,959 in the classification dataset and 137,743 in the localization dataset acquired at different times of the day and different periods of the year by thousands of traffic cameras deployed all over Canada and the United States. Those images have been selected to cover a wide range of challenges and are representative of typical visual data captured today in urban traffic scenarios. Each moving object has been carefully identified by nearly 200 persons to enable a quantitative comparison and ranking of various algorithms. This dataset aims to provide a rigorous benchmarking facility for training and testing existing and new algorithms for the classification and localization of moving vehicles in traffic scenes. The dataset is divided in two parts : the “classification challenge dataset” and the “localization challenge dataset”.

### Credits
Z. Luo, F.B.Charron, C.Lemaire, J.Konrad, S.Li, A.Mishra, A. Achkar, J. Eichel, P-M Jodoin MIO-TCD: A new benchmark dataset for vehicle classification and localization in press at IEEE Transactions on Image Processing, 2018

### Classification challenge dataset
Contains 648,959 images divided into 11 categories: Articulated truck Bicycle Bus Car Motorcycle Non-motorized vehicle Pedestrian Pickup truck Single unit truck Work van Background

##### Goal
The goal of this challenge is to correctly label each image

### Localization challenge dataset
Contains 137,743 high-resolution images containing one (or more) foreground object(s) with one of the following 11 labels: Articulated truck Bicycle Bus Car Motorcycle Motorized vehicle (i.e. Vehicles that are too small to be labeled into a specific category) Non-motorized vehicle Pedestrian Pickup truck Single unit truck Work van

##### Goal
The goal of this challenge is to correctly localize and classify each foreground object.

# ECSE415-Project

The project report contains the written answers that we will submit, as well as a prototype of our algorithms. The code that does the real work will be in the `MIO-TCD-Classification-Code` and `MIO-TCD-Localization-Code` folders. Scripts that handle the parsing the image files and writing the result to csv are already there (from the website). We just need to call our classifier in the correct place.

In the two folders mentioned above are also csv files. `gt-train.csv` is the **G**round **T**ruth, and we will write our results into `your_results.csv`.
