# Resnet-MLP

This idea is inspired by the [PNI paper](https://arxiv.org/pdf/2211.12634v3.pdf) and starts with a very
simplified model:
The intermediate outputs of the ResNet are put into an MLP network with a binary output of
(correct, anomaly).

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