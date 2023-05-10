"""Inference Entrypoint script."""

from flask import Flask
from flask import render_template
import torch
import requests

from argparse import ArgumentParser, Namespace
from pathlib import Path
from dataclasses import dataclass

from pytorch_lightning import Trainer, seed_everything
from torch.utils.data import DataLoader

from anomalib.config import get_configurable_parameters
from anomalib.data import get_datamodule
from anomalib.data.utils import TestSplitMode
from anomalib.models import get_model
from anomalib.utils.callbacks import LoadModelCallback, get_callbacks
from anomalib.utils.loggers import configure_logger, get_experiment_logger
from anomalib.data.mvtec import MVTec

import os
from os import environ 

from datetime import datetime
import numpy


app = Flask(__name__)

dataset_path = Path(os.path.join(app.static_folder, environ.get("DATASET_PATH")))
config_path = Path("./config-generic.yaml")

trainer = None
patchcore = None
datamodule = None

@dataclass
class TestResults:
    pixel_F1Score: float
    pixel_AUROC: float
    image_F1Score: float
    image_AUROC: float

@app.route('/')
def index(name=None):
    images = [os.path.join(environ.get("DATASET_PATH"), "grid/train/good", image) for image in os.listdir(os.path.join(app.static_folder, environ.get("DATASET_PATH"), "grid/train/good"))]
    print(images)
    return render_template('index.html', name=name, input_images=images)

@app.route('/load-server-metadata')
def metadata(name=None):
    metadata = requests.get("http://localhost:8080/metadata").json()
    print(metadata)
    return render_template('fragments/server-bank.html', metadata=metadata)

@app.route('/load-server-bank')
def load_bank(name=None):
    global patchcore

    bank_json = requests.get("http://localhost:8080/get-bank").json()
    memory_bank = torch.from_numpy(numpy.asarray(bank_json)).float()
    print(f"received memory bank with shape {memory_bank.size()}: {memory_bank}")
    patchcore.model.memory_bank = memory_bank
    return render_template('fragments/local-bank.html', shape=memory_bank.size(), time=datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

@app.route('/send')
def send(name=None):
    send_local_memory_bank()
    return {}

@app.route('/train')
def train(name=None):
    (shape, time) = anomalib_train()
    return render_template('fragments/local-bank.html', shape=shape, time=time)

@app.route('/test')
def test(name=None):
    results = anomalib_test()
    anomaly_types = ["bent", "broken", "glue", "metal_contamination", "thread", "good"]
    samples = ["000.png", "001.png"]
    images = [os.path.join("results", anomaly, sample) for sample in samples for anomaly in anomaly_types]
    return render_template('fragments/test-results.html', results=results, images=images)

def send_local_memory_bank():
    global patchcore

    # patchcore = model.load_from_checkpoint(str(Path("results/partial2/patchcore/mvtec/grid/run/weights/model.ckpt")))
    print(f"sending memory bank with shape {patchcore.model.memory_bank.size()}: {patchcore.model.memory_bank}")
    to_numpy = patchcore.model.memory_bank.cpu().data.numpy()
    requests.post("http://localhost:8080/memory-bank", json=to_numpy.tolist())

def load():
    global trainer
    global patchcore
    global datamodule

    print(config_path)
    config = get_configurable_parameters(model_name="patchcore", config_path=config_path)
    if config.project.get("seed") is not None:
        seed_everything(config.project.seed)

    datamodule = MVTec(
            root=dataset_path,
            category=config.dataset.category,
            image_size=(config.dataset.image_size[0], config.dataset.image_size[1]),
            center_crop=(config.dataset.get("center_crop")[0], config.dataset.get("center_crop")[1]),
            normalization=config.dataset.normalization,
            train_batch_size=config.dataset.train_batch_size,
            eval_batch_size=config.dataset.eval_batch_size,
            num_workers=config.dataset.num_workers,
            task=config.dataset.task,
            transform_config_train=config.dataset.transform_config.train,
            transform_config_eval=config.dataset.transform_config.eval,
            test_split_mode=config.dataset.test_split_mode,
            test_split_ratio=config.dataset.test_split_ratio,
            val_split_mode=config.dataset.val_split_mode,
            val_split_ratio=config.dataset.val_split_ratio,
        )

    patchcore = get_model(config)
    experiment_logger = get_experiment_logger(config)
    callbacks = get_callbacks(config)

    trainer = Trainer(**config.trainer, logger=experiment_logger, callbacks=callbacks)

    print(trainer.callbacks)

def anomalib_train():
    global patchcore
    
    trainer.fit(model=patchcore, datamodule=datamodule)

    load_model_callback = LoadModelCallback(weights_path=trainer.checkpoint_callback.best_model_path)
    trainer.callbacks.insert(0, load_model_callback)  # pylint: disable=no-member

    return (patchcore.model.memory_bank.size(), datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

def anomalib_test():
    global patchcore
    # ImageVisualizerCallback(mode="full", task=task, image_save_path="./results/images")
    test_results = trainer.test(model=patchcore, datamodule=datamodule)

    return TestResults(test_results[0]["pixel_F1Score"], test_results[0]["pixel_AUROC"], test_results[0]["image_F1Score"], test_results[0]["image_AUROC"])

load()