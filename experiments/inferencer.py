import torch
from omegaconf import ListConfig
from torch.types import Number

from config import get_configurable_parameters
from data.task_type import TaskType
from data.utils import InputNormalizationMethod, get_transforms
from data.utils.boxes import masks_to_boxes
from models import get_model
from models.components.base import AnomalyModule
from abc import ABC
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from omegaconf import DictConfig
from skimage.morphology import dilation
from skimage.segmentation import find_boundaries
from torch import Tensor

from data.utils import read_image
from post_processing import ImageResult, compute_mask
from post_processing.normalization.cdf import normalize as normalize_cdf
from post_processing.normalization.cdf import standardize
from post_processing.normalization.min_max import normalize as normalize_min_max


def _get_model_metadata(model: AnomalyModule) -> dict[str, Tensor]:
    """Get meta data related to normalization from model.

    Args:
        model (AnomalyModule): Anomaly model which contains metadata related to normalization.

    Returns:
        dict[str, Tensor]: Model metadata
    """
    metadata = {}
    cached_metadata: dict[str, Number | Tensor] = {
        "image_threshold": model.image_threshold.cpu().value.item(),
        "pixel_threshold": model.pixel_threshold.cpu().value.item(),
    }
    if hasattr(model, "normalization_metrics") and model.normalization_metrics.state_dict() is not None:
        for key, value in model.normalization_metrics.state_dict().items():
            cached_metadata[key] = value.cpu()
    # Remove undefined values by copying in a new dict
    for key, val in cached_metadata.items():
        if not np.isinf(val).all():
            metadata[key] = val
    del cached_metadata
    return metadata


class TorchInferencer(ABC):
    """PyTorch implementation for the inference.

    Args:
        config (str | Path | DictConfig | ListConfig): Configurable parameters that are used
            during the training stage.
        model_source (str | Path | AnomalyModule): Path to the model ckpt file or the Anomaly model.
        metadata_path (str | Path, optional): Path to metadata file. If none, it tries to load the params
                from the model state_dict. Defaults to None.
        device (str | None, optional): Device to use for inference. Options are auto, cpu, cuda. Defaults to "auto".
    """

    def __init__(
        self,
        config: str | Path | DictConfig | ListConfig,
        model_source: str | Path | AnomalyModule,
        device: str = "auto",
    ) -> None:
        self.device = self._get_device(device)

        # Check and load the configuration
        if isinstance(config, (str, Path)):
            self.config = get_configurable_parameters(config_path=config)
        elif isinstance(config, (DictConfig, ListConfig)):
            self.config = config
        else:
            raise ValueError(f"Unknown config type {type(config)}")

        # Check and load the model weights.
        if isinstance(model_source, AnomalyModule):
            self.model = model_source
        else:
            self.model = self.load_model(model_source)

        self.metadata: dict[str, float | np.ndarray | Tensor] | DictConfig
        self.metadata = _get_model_metadata(self.model)

    def predict(
        self,
        image: str | Path | np.ndarray,
        metadata: dict[str, Any] | None = None,
    ):
        """Perform a prediction for a given input image.

        The main workflow is (i) pre-processing, (ii) forward-pass, (iii) post-process.

        Args:
            image (Union[str, np.ndarray]): Input image whose output is to be predicted.
                It could be either a path to image or numpy array itself.

            metadata: Metadata information such as shape, threshold.

        Returns:
            ImageResult: Prediction results to be visualized.
        """
        if metadata is None:
            if hasattr(self, "metadata"):
                metadata = getattr(self, "metadata")
            else:
                metadata = {}
        if isinstance(image, (str, Path)):
            image_arr: np.ndarray = read_image(image)
        else:  # image is already a numpy array. Kept for mypy compatibility.
            image_arr = image
        metadata["image_shape"] = image_arr.shape[:2]

        processed_image = self.pre_process(image_arr)
        predictions = self.forward(processed_image)
        output = self.post_process(predictions, metadata=metadata)

        if isinstance(predictions, Tensor):
            anomaly_map = predictions.detach().cpu().numpy()
            pred_score = anomaly_map.reshape(-1).max()
        else:
            # NOTE: Patchcore `forward`` returns heatmap and score.
            #   We need to add the following check to ensure the variables
            #   are properly assigned. Without this check, the code
            #   throws an error regarding type mismatch torch vs np.
            if isinstance(predictions[1], (Tensor)):
                anomaly_map, pred_score = predictions
                anomaly_map = anomaly_map.detach().cpu().numpy()
                pred_score = pred_score.detach().cpu().numpy()
            else:
                anomaly_map, pred_score = predictions
                pred_score = pred_score.detach()

        return pred_score, output["pred_label"]

    @staticmethod
    def _superimpose_segmentation_mask(metadata: dict, anomaly_map: np.ndarray, image: np.ndarray) -> np.ndarray:
        """Superimpose segmentation mask on top of image.

        Args:
            metadata (dict): Metadata of the image which contains the image size.
            anomaly_map (np.ndarray): Anomaly map which is used to extract segmentation mask.
            image (np.ndarray): Image on which segmentation mask is to be superimposed.

        Returns:
            np.ndarray: Image with segmentation mask superimposed.
        """
        pred_mask = compute_mask(anomaly_map, 0.5)  # assumes predictions are normalized.
        image_height = metadata["image_shape"][0]
        image_width = metadata["image_shape"][1]
        pred_mask = cv2.resize(pred_mask, (image_width, image_height))
        boundaries = find_boundaries(pred_mask)
        outlines = dilation(boundaries, np.ones((7, 7)))
        image[outlines] = [255, 0, 0]
        return image

    def __call__(self, image: np.ndarray) -> ImageResult:
        """Call predict on the Image.

        Args:
            image (np.ndarray): Input Image

        Returns:
            ImageResult: Prediction results to be visualized.
        """
        return self.predict(image)

    @staticmethod
    def _normalize(
        pred_scores: Tensor | np.float32,
        metadata: dict | DictConfig,
        anomaly_maps: Tensor | np.ndarray | None = None,
    ) -> tuple[np.ndarray | Tensor | None, float]:
        """Applies normalization and resizes the image.

        Args:
            pred_scores (Tensor | np.float32): Predicted anomaly score
            metadata (dict | DictConfig): Meta data. Post-processing step sometimes requires
                additional meta data such as image shape. This variable comprises such info.
            anomaly_maps (Tensor | np.ndarray | None): Predicted raw anomaly map.

        Returns:
            tuple[np.ndarray | Tensor | None, float]: Post processed predictions that are ready to be
                visualized and predicted scores.
        """

        # min max normalization
        if "min" in metadata and "max" in metadata:
            if anomaly_maps is not None:
                anomaly_maps = normalize_min_max(
                    anomaly_maps,
                    metadata["pixel_threshold"],
                    metadata["min"],
                    metadata["max"],
                )
            pred_scores = normalize_min_max(
                pred_scores,
                metadata["image_threshold"],
                metadata["min"],
                metadata["max"],
            )

        # standardize pixel scores
        if "pixel_mean" in metadata.keys() and "pixel_std" in metadata.keys():
            if anomaly_maps is not None:
                anomaly_maps = standardize(
                    anomaly_maps, metadata["pixel_mean"], metadata["pixel_std"], center_at=metadata["image_mean"]
                )
                anomaly_maps = normalize_cdf(anomaly_maps, metadata["pixel_threshold"])

        # standardize image scores
        if "image_mean" in metadata.keys() and "image_std" in metadata.keys():
            pred_scores = standardize(pred_scores, metadata["image_mean"], metadata["image_std"])
            pred_scores = normalize_cdf(pred_scores, metadata["image_threshold"])

        return anomaly_maps, float(pred_scores)

    @staticmethod
    def _get_device(device: str) -> torch.device:
        """Get the device to use for inference.

        Args:
            device (str): Device to use for inference. Options are auto, cpu, cuda.

        Returns:
            torch.device: Device to use for inference.
        """
        if device not in ("auto", "cpu", "cuda", "gpu"):
            raise ValueError(f"Unknown device {device}")

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "gpu":
            device = "cuda"
        return torch.device(device)

    def load_model(self, path: str | Path) -> AnomalyModule:
        """Load the PyTorch model.

        Args:
            path (str | Path): Path to model ckpt file.

        Returns:
            (AnomalyModule): PyTorch Lightning model.
        """
        model = get_model(self.config)
        model.load_state_dict(torch.load(path, map_location=self.device)["state_dict"])
        model.eval()
        return model.to(self.device)

    def pre_process(self, image: np.ndarray) -> Tensor:
        """Pre-process the input image by applying transformations.

        Args:
            image (np.ndarray): Input image

        Returns:
            Tensor: pre-processed image.
        """
        transform_config = (
            self.config.dataset.transform_config.eval if "transform_config" in self.config.dataset.keys() else None
        )

        image_size = (self.config.dataset.image_size[0], self.config.dataset.image_size[1])
        center_crop = self.config.dataset.get("center_crop")
        if center_crop is not None:
            center_crop = tuple(center_crop)
        normalization = InputNormalizationMethod(self.config.dataset.normalization)
        transform = get_transforms(
            config=transform_config, image_size=image_size, center_crop=center_crop, normalization=normalization
        )
        processed_image = transform(image=image)["image"]

        if len(processed_image) == 3:
            processed_image = processed_image.unsqueeze(0)

        return processed_image.to(self.device)

    def forward(self, image: Tensor) -> Tensor:
        """Forward-Pass input tensor to the model.

        Args:
            image (Tensor): Input tensor.

        Returns:
            Tensor: Output predictions.
        """
        return self.model(image)

    def post_process(self, predictions: Tensor, metadata: dict | DictConfig | None = None) -> dict[str, Any]:
        """Post process the output predictions.

        Args:
            predictions (Tensor): Raw output predicted by the model.
            metadata (dict, optional): Meta data. Post-processing step sometimes requires
                additional meta data such as image shape. This variable comprises such info.
                Defaults to None.

        Returns:
            dict[str, str | float | np.ndarray]: Post processed prediction results.
        """
        if metadata is None:
            metadata = self.metadata

        if isinstance(predictions, Tensor):
            anomaly_map = predictions.detach().cpu().numpy()
            pred_score = anomaly_map.reshape(-1).max()
        else:
            # NOTE: Patchcore `forward`` returns heatmap and score.
            #   We need to add the following check to ensure the variables
            #   are properly assigned. Without this check, the code
            #   throws an error regarding type mismatch torch vs np.
            if isinstance(predictions[1], (Tensor)):
                anomaly_map, pred_score = predictions
                anomaly_map = anomaly_map.detach().cpu().numpy()
                pred_score = pred_score.detach().cpu().numpy()
            else:
                anomaly_map, pred_score = predictions
                pred_score = pred_score.detach()

        # Common practice in anomaly detection is to assign anomalous
        # label to the prediction if the prediction score is greater
        # than the image threshold.
        pred_label: str | None = None
        if "image_threshold" in metadata:
            # import ipdb; ipdb.set_trace()
            pred_idx = pred_score >= metadata["image_threshold"]
            pred_label = "Anomalous" if pred_idx else "Normal"
            # print("Setting prediction label to", pred_label)
            # print("Prediction score:", pred_score)
            # print("Image threshold:", metadata["image_threshold"])

        pred_mask: np.ndarray | None = None
        if "pixel_threshold" in metadata:
            pred_mask = (anomaly_map >= metadata["pixel_threshold"]).squeeze().astype(np.uint8)

        anomaly_map = anomaly_map.squeeze()
        anomaly_map, pred_score = self._normalize(anomaly_maps=anomaly_map, pred_scores=pred_score, metadata=metadata)

        if isinstance(anomaly_map, Tensor):
            anomaly_map = anomaly_map.detach().cpu().numpy()

        if "image_shape" in metadata and anomaly_map.shape != metadata["image_shape"]:
            image_height = metadata["image_shape"][0]
            image_width = metadata["image_shape"][1]
            anomaly_map = cv2.resize(anomaly_map, (image_width, image_height))

            if pred_mask is not None:
                pred_mask = cv2.resize(pred_mask, (image_width, image_height))

        if self.config.dataset.task == TaskType.DETECTION:
            pred_boxes = masks_to_boxes(torch.from_numpy(pred_mask))[0][0].numpy()
            box_labels = np.ones(pred_boxes.shape[0])
        else:
            pred_boxes = None
            box_labels = None

        return {
            "anomaly_map": anomaly_map,
            "pred_label": pred_label,
            "pred_score": pred_score,
            "pred_mask": pred_mask,
            "pred_boxes": pred_boxes,
            "box_labels": box_labels,
        }
