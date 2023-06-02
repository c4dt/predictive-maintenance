# Predictive maintenance

This experiment attempts to do a transfer learning from PatchCore to Linear Regression.

The goal is to see if the features we got from ResNet and the Patch scores we get from PatchCore,
can be used to train a linear regression model that would be able to predict the anomaly score of
a new patch, thus learning to see if an image is anomalous or not.

## Implementation
The code you see here is only for the training of the linear regression model. The code for the
PatchCore learning from PatchCore will be added later. 

But for now, we assume we got the dataset (./dataset_patchscores.json) from PatchCore.

## Dataset
For now, the dataset is on Ahmed's machine (the file is 5 Gbytes - too big for github).
Further development will allows for either downloading it or generating it from PatchCore.


## Blockers
- We need the anomaly score threshold from PatchCore to be able to generate a binary classification.

## Next step:
- Add the code that trains the PatchCore model instead of directly adding the dataset.