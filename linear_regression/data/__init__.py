import logging

from omegaconf import DictConfig, ListConfig

from .base.datamodule import AnomalibDataModule
from .base.dataset import AnomalibDataset
from .mvtec import MVTec

logger = logging.getLogger(__name__)


def get_datamodule(config: DictConfig | ListConfig) -> AnomalibDataModule:
    """Get Anomaly Datamodule.

    Args:
        config (DictConfig | ListConfig): Configuration of the anomaly model.

    Returns:
        PyTorch Lightning DataModule
    """
    logger.info("Loading the datamodule")

    # convert center crop to tuple
    center_crop = config.dataset.get("center_crop")
    if center_crop is not None:
        center_crop = (center_crop[0], center_crop[1])

    return MVTec(
        root=config.dataset.path,
        category=config.dataset.category,
        image_size=(config.dataset.image_size[0], config.dataset.image_size[1]),
        center_crop=center_crop,
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
