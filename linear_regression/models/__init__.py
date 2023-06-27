import os

from omegaconf import DictConfig, ListConfig
from torch import load


from models.patchcore import Patchcore, PatchcoreLightning


def get_model(config: DictConfig | ListConfig) -> PatchcoreLightning:
    """Load model from the configuration file.

    Args:
        config (DictConfig | ListConfig): Config.yaml loaded using OmegaConf

    Returns:
        PatchcoreLightning
    """

    model = PatchcoreLightning(config)

    if "init_weights" in config.keys() and config.init_weights:
        model.load_state_dict(load(os.path.join(config.project.path, config.init_weights))["state_dict"], strict=False)

    return model
