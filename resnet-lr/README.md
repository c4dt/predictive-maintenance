# Resnet-LinearRegression

This idea is inspired by the [PNI paper](https://arxiv.org/pdf/2211.12634v3.pdf) and starts with a very
simplified model:

The code in here has been proposed by ChatGPT and fixed by it, too.
It is able to use `CUDA` or `mps` (mac apple silicon accelerator) for acceleration. 

## Algorithm
This algorithm uses Patchcore to update the dataset with an anomaly score per item in dataset.
Once we have a score per item in dataset, we use that score to train a linear regression model 
The intermediate outputs of the ResNet are put into an MLP network with a binary output of
(correct, anomaly).

## Architecture
- patchcore is included as an installable component

## Steps followed:
1. Train the PatchCore algorithm on the MVtec dataset (half the training dataset later)
2. Check the test accuracy is good
3. Use the inference part to only give the anomaly score per object (a list of numbers) -> apply to all dataset
4. Pass the data again, along with their score to the regression model.
5. Train the regression model on this data, and test it on the training dataset (should generate a binary model, or an anomaly score as well)

## Installation

Tested with python 3.10.11

Setup env:

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

For Apple Silicon, follow the instructions here: https://developer.apple.com/metal/pytorch/

## Running it

Once all the dependencies are installed, you can run it with

```bash
python train.py
```

It will print out the device it found for training:
- `cpu` - slowest - never let it run through, but probably at least 1h
- `cuda` - don't have it
- `mps` - for Apple Silicon - on an M2 it takes about 10 minutes

## Dataset
This model uses MVTec AD dataset.  
the dataset can be downloaded here: https://www.mvtec.com/company/research/datasets/mvtec-ad

## Observations
After training the model on the MVTec data, we realize that the MVTec dataset is designed
only for models that use a single class (normal data / non-anomalous).

That's why the result we saw is that the resulting model classified all data as normal
as it's the only class it was trained on.
