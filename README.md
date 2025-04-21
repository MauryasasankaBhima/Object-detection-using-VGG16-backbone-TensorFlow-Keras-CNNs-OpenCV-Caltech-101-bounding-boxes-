# 
 End-to-End Deep Learning Pipeline for Object Detection and Classification
This project implements a comprehensive deep learning pipeline for object detection and classification using the Caltech-101 dataset. The pipeline covers data preprocessing, model development with a pre-trained VGG16 backbone, training, evaluation, and visualization of predictions. 

Objective
To build and evaluate a deep learning model that performs simultaneous object detection (bounding box regression) and classification using transfer learning with VGG16, optimizing for accuracy and bounding box precision.

 Tools & Techniques



Task
Tools / Libraries



Data Manipulation
pandas, NumPy


Preprocessing
LabelBinarizer, Image Resizing, Normalization


Modeling
VGG16, tensorflow.keras (Dense, Dropout layers)


Model Evaluation
Classification Accuracy, Mean Squared Error (MSE) for Bounding Boxes


Visualization
matplotlib, opencv-python (for bounding box visualization)



 Workflow Overview

Data Preparation

Loaded the Caltech-101 dataset with images and ground truth (gt_summary.csv).
Filtered for three classes: airplanes, faces, motorbikes.
Preprocessed images by resizing to 224x224 and normalizing pixel values (0-1).
Normalized bounding box coordinates (0-1) and one-hot encoded class labels.
Split data into training (75%), validation (15%), and test (10%) sets.


Model Development

Used pre-trained VGG16 (ImageNet weights) as the backbone, with frozen layers.
Added custom heads:
Classification Head: Dense layers with softmax for class prediction.
**Regression  | tensorflow.keras (Dense, Dropout layers)            |


Compiled with categorical cross-entropy (classification) and MSE (regression) losses.


Model Training

Trained the model for 15 epochs using the Adam optimizer (learning rate 1e-4).
Evaluated on validation data during training.


Prediction and Visualization

Predicted bounding boxes and class labels on test images.
Visualized ground truth (green) and predicted (blue) bounding boxes with class labels.


Performance Evaluation

Monitored classification accuracy and bounding box MSE on validation data.
Generated visualizations for qualitative assessment of predictions.




Sample Evaluation Metrics




Metric
Training Value
Validation Value


Bounding Box MSE
1.62e-4
7.38e-4


Total Loss
4.20e-4
7.54e-4



Key Highlights

Built an end-to-end deep learning pipeline for object detection and classification.
Leveraged transfer learning with VGG16 for efficient feature extraction.
Achieved near-perfect classification accuracy and low bounding box MSE.
Visualized predictions to compare ground truth and predicted bounding boxes.
Processed and normalized image data for robust model training.





