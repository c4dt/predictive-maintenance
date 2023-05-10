from flask import Flask
from flask import request
import numpy
import torch

from pathlib import Path

from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from anomalib.config import get_configurable_parameters
from anomalib.data.inference import InferenceDataset
from anomalib.data.utils import InputNormalizationMethod, get_transforms
from anomalib.models import get_model
from anomalib.utils.callbacks import get_callbacks

import os
from os.path import exists
from datetime import datetime
from dataclasses import dataclass
import requests

app = Flask(__name__)

# Buffer of received memory banks
memory_banks = []

# Number of received banks before merge
BUFFER_SIZE = 1

model_path = str(Path("./server-merged.ckpt"))
config = None
patchcore = None

@dataclass
class Metadata:
    model_size: float
    last_update: str
    shape: torch.Size

metadata = Metadata(0, None, None)


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/metadata")
def metadata():
    return metadata.__dict__

@app.route("/get-bank")
def get_bank():
    global patchcore

    # patchcore = model.load_from_checkpoint(str(Path("results/partial2/patchcore/mvtec/grid/run/weights/model.ckpt")))
    print(f"sending memory bank with shape {patchcore.model.memory_bank.size()}: {patchcore.model.memory_bank}")
    to_numpy = patchcore.model.memory_bank.cpu().data.numpy()
    return to_numpy.tolist()


@app.route("/memory-bank", methods=['POST'])
def memory_bank():
    memory_bank = torch.from_numpy(numpy.asarray(request.json))
    print(f"received memory bank with shape {memory_bank.size()}: {memory_bank}")
    print("adding to current buffer...")
    memory_banks.append(memory_bank)

    remaining_banks = BUFFER_SIZE - len(memory_banks)
    print(f"buffer size now {len(memory_banks)}. Missing {remaining_banks} memory banks before merge")

    if remaining_banks <= 0:
        merge()
    return {}

def load():
    global patchcore
    global config

    config = get_configurable_parameters(config_path=Path("config-partial1.yaml"))
    model = get_model(config)

    if exists(model_path):
        patchcore = model.load_from_checkpoint(model_path)
    else:
        patchcore = model.load_from_checkpoint(str(Path("./server-merged-initial.ckpt")))

    update_metadata()

def update_metadata():
    if exists(model_path):
        metadata.model_size = round(os.path.getsize(model_path)/(pow(1024,2)), 2)
    else:
        metadata.model_size = round(os.path.getsize(str(Path("./server-merged-initial.ckpt")))/(pow(1024,2)), 2)
    
    metadata.last_update = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    metadata.shape = patchcore.model.memory_bank.size()

def merge():
    global memory_banks
    for bank in memory_banks:
        print(bank.size())

    

    merged_memory_bank = torch.cat([patchcore.model.memory_bank] + memory_banks)
    print(f"Merged memory bank shape:{merged_memory_bank.size()}")

    # Hacky way of saving the model with Lightning without training
    patchcore.model.memory_bank = merged_memory_bank
    callbacks = get_callbacks(config)
    trainer = Trainer(callbacks=callbacks, **config.trainer)
    trainer.strategy.connect(patchcore)
    trainer.save_checkpoint(model_path)

    update_metadata()

    # Reset banks
    memory_banks = []

load()