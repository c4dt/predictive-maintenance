"""Pre Process.

This module contains `PreProcessor` class that applies preprocessing
to an input image before the forward-pass stage.
"""
import logging
import warnings
from typing import Any

import albumentations as A
from albumentations.pytorch import ToTensorV2

logger = logging.getLogger(__name__)


def get_image_height_and_width(image_size: int | tuple | None = None) -> tuple[int | None, int | None]:
    """Get image height and width from ``image_size`` variable.

    Args:
        image_size (int | tuple | None, optional): Input image size.

    Raises:
        ValueError: Image size not None, int or tuple.

    Examples:
        >>> get_image_height_and_width(image_size=256)
        (256, 256)

        >>> get_image_height_and_width(image_size=(256, 256))
        (256, 256)

        >>> get_image_height_and_width(image_size=(256, 256, 3))
        (256, 256)

        >>> get_image_height_and_width(image_size=256.)
        Traceback (most recent call last):
        File "<string>", line 1, in <module>
        File "<string>", line 18, in get_image_height_and_width
        ValueError: ``image_size`` could be either int or tuple[int, int]

    Returns:
        tuple[int | None, int | None]: A tuple containing image height and width values.
    """
    height_and_width: tuple[int | None, int | None]
    if isinstance(image_size, int):
        height_and_width = (image_size, image_size)
    elif isinstance(image_size, tuple):
        height_and_width = int(image_size[0]), int(image_size[1])
    elif image_size is None:
        height_and_width = (None, None)
    else:
        raise ValueError("``image_size`` could be either int or tuple[int, int]")

    return height_and_width


def get_transforms(
    config: str | A.Compose | None = None,
    image_size: int | tuple | None = None,
    to_tensor: bool = True,
) -> A.Compose:
    """Get transforms from config or image size.

    Args:
        config (str | A.Compose | None, optional): Albumentations transforms.
            Either config or albumentations ``Compose`` object. Defaults to None.
        image_size (int | tuple | None, optional): Image size to transform. Defaults to None.
        to_tensor (bool, optional): Boolean to convert the final transforms into Torch tensor. Defaults to True.

    Raises:
        ValueError: When both ``config`` and ``image_size`` is ``None``.
        ValueError: When ``config`` is not a ``str`` or `A.Compose`` object.

    Returns:
        A.Compose: Albumentation ``Compose`` object containing the image transforms.

    Examples:
        >>> import skimage
        >>> image = skimage.data.astronaut()

        >>> transforms = get_transforms(image_size=256, to_tensor=False)
        >>> output = transforms(image=image)
        >>> output["image"].shape
        (256, 256, 3)

        >>> transforms = get_transforms(image_size=256, to_tensor=True)
        >>> output = transforms(image=image)
        >>> output["image"].shape
        torch.Size([3, 256, 256])


        Transforms could be read from albumentations Compose object.
        >>> import albumentations as A
        >>> from albumentations.pytorch import ToTensorV2
        >>> config = A.Compose([A.Resize(512, 512), ToTensorV2()])
        >>> transforms = get_transforms(config=config, to_tensor=False)
        >>> output = transforms(image=image)
        >>> output["image"].shape
        (512, 512, 3)
        >>> type(output["image"])
        numpy.ndarray

        Transforms could be deserialized from a yaml file.
        >>> transforms = A.Compose([A.Resize(1024, 1024), ToTensorV2()])
        >>> A.save(transforms, "/tmp/transforms.yaml", data_format="yaml")
        >>> transforms = get_transforms(config="/tmp/transforms.yaml")
        >>> output = transforms(image=image)
        >>> output["image"].shape
        torch.Size([3, 1024, 1024])
    """
    warnings.warn(
        DeprecationWarning(
            "The function pre_processing.pre_process.get_transforms is deprecated and will be removed in a "
            "future release. Please use data.utils.transform.get_transforms instead."
        )
    )

    if config is None is image_size:
        raise ValueError(
            "Both config and image_size cannot be `None`. "
            "Provide either config file to de-serialize transforms "
            "or image_size to get the default transformations"
        )

    transforms: A.Compose

    if config is None and image_size is not None:
        logger.warning("Transform configs has not been provided. Images will be normalized using ImageNet statistics.")

        height, width = get_image_height_and_width(image_size)
        transforms = A.Compose(
            [
                A.Resize(height=height, width=width, always_apply=True),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

    if config is not None:
        if isinstance(config, str):
            transforms = A.load(filepath=config, data_format="yaml")
        elif isinstance(config, A.Compose):
            transforms = config
        else:
            raise ValueError("config could be either ``str`` or ``A.Compose``")

    if not to_tensor:
        if isinstance(transforms[-1], ToTensorV2):
            transforms = A.Compose(transforms[:-1])

    # always resize to specified image size
    if not any(isinstance(transform, A.Resize) for transform in transforms) and image_size is not None:
        height, width = get_image_height_and_width(image_size)
        transforms = A.Compose([A.Resize(height=height, width=width, always_apply=True), transforms])

    return transforms
