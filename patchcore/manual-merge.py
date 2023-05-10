"""Inference Entrypoint script."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch

from argparse import ArgumentParser, Namespace
from pathlib import Path

from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from anomalib.config import get_configurable_parameters
from anomalib.data.inference import InferenceDataset
from anomalib.data.utils import InputNormalizationMethod, get_transforms
from anomalib.models import get_model
from anomalib.utils.callbacks import get_callbacks


def get_args() -> Namespace:
    """Get command line arguments.

    Returns:
        Namespace: List of arguments.
    """
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, required=True, help="Path to a config file")
    parser.add_argument('--weights', nargs='+', type=Path, required=True, help="Paths to models weights")

    args = parser.parse_args()
    return args


def infer():
    """Run inference."""
    args = get_args()
    config = get_configurable_parameters(config_path=args.config)

    # create model and trainer
    model = get_model(config)

    # Merging
    patchcores = []
    for weight in args.weights:
        patchcores.append(model.load_from_checkpoint(str(weight)))

    for patchcore in patchcores:
        print(patchcore.model.memory_bank.size())

    merged_memory_bank = torch.cat([patchcore.model.memory_bank for patchcore in patchcores])
    print(f"Merged memory bank shape:{merged_memory_bank.size()}")

    # Hacky way of saving the model with Lightning

    patchcores[0].model.memory_bank = merged_memory_bank
    callbacks = get_callbacks(config)
    trainer = Trainer(callbacks=callbacks, **config.trainer)
    trainer.strategy.connect(patchcores[0])
    trainer.save_checkpoint("./merged.ckpt")

    # Reloading to check memory bank size

    print(str(Path("./merged.ckpt")))
    mergedModel = model.load_from_checkpoint(str(Path("./merged.ckpt")))
    print(mergedModel.model.memory_bank.size())


if __name__ == "__main__":
    infer()
