# Predictive maintenance (Linear Regression)

This experiment attempts to do a transfer learning from PatchCore to Linear Regression.

The goal is to see if the features we got from ResNet and the Patch scores we get from PatchCore,
can be used to train a linear regression model that would be able to predict the anomaly score of
a new patch, thus learning to see if an image is anomalous or not.

## Setup and Running
1. Install the requirements (Python 3.10)
2. Prepare Datasets
2. Use the `linear_regression.ipynb` notebook to train the model and evaluate it.


### Dataset
Download MVTec here: https://www.mvtec.com/company/research/datasets/mvtec-ad/

For this experiment we will use the grid dataset.
Divide the grid dataset into datasets (a) and (b).

(You can use the notebook as a guide to do this separation.)

Use half the dataset to train the PatchCore model.
Store it in `datasets/MVTec_a/grid`  
and b: only use the second half `datasets/MVTec_b/grid` to train the LR model.

(We only use the grid dataset is you have noticed)

### Implementation
the file `linear_regression.ipynb` contains the code along with all the instructions needed to run 
the whole solution from first training MVTec to on hal the dataset to evaluating the LR model at the end.
