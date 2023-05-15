# Resnet-LR

This idea is inspired by the [PNI paper](https://arxiv.org/pdf/2211.12634v3.pdf) and starts with a very
simplified model:
The intermediate outputs of the ResNet are put into a Linear Regression model
which outputs a number between 0, and $\inf$


The code in here has been proposed by ChatGPT and fixed by it, too.
It is able to use `CUDA` or `mps` (mac apple silicon accelerator) for acceleration. 

## Installation

Tested with python 3.10.9

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
The model runs by default on the CIFAR10 dataset, and downloads it if it's not already downloaded.
To change the model to MVTec, simply, go to the `train.py` file, and change the line
```main(device, mvtec=False)```

to ```main(device, mvtec=True)```
However, you need to download the dataset yourself, reference its location in `BinaryMVtec.__init__.root`.
Note that the MVTec dataset comes with different datasets as sub-folders, make sure that your root directory
refers to a specific subset. 

## Observations
After training the model on the MVTec data, we realize that the MVTec dataset is designed
only for models that use a single class (normal data / non-anomalous).

That's why the result we saw is that the resulting model classified all data as normal
as it's the only class it was trained on.
