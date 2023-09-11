# Predictive maintenance (Linear Regression)

This experiment attempts to do a transfer learning from PatchCore to Linear Regression.

The goal is to see if the features we got from ResNet and the Patch scores we get from PatchCore,
can be used to train a linear regression model that would be able to predict the anomaly score of
a new patch, thus learning to see if an image is anomalous or not.

## Setup and Running
1. Install the requirements (from requirements.txt) (Python 3.10)
2. Prepare Datasets
2. Use the different notebooks to train the contained model and evaluate it.


### Dataset
Download MVTec here: https://www.mvtec.com/company/research/datasets/mvtec-ad/

For this experiment we will only use the **grid** dataset.
Divide the grid dataset into datasets (a) and (b).
The test dataset should be duplicated in both **dataset a** and **dataset b**

(You can use the notebook as a guide to do this separation.)

Use half the dataset to train the PatchCore model.
Store it in `datasets/MVTec_a/grid`  
and b: only use the second half `datasets/MVTec_b/grid` to train the LR model.

### Implementation
This project is a modified version of the library `anomalib` with the addition of our own experiments written
in the multiple jupyter notebooks available.  
The notebooks have very similar code, but each of them represent a different experiment conditions.  
The notebooks also contain instructions on how to run them.
