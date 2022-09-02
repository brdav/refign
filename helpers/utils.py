import os

import pytorch_lightning as pl
from PIL import Image

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def resolve_ckpt_dir(trainer: pl.Trainer):
    # From pytorch_lightning.model_checkpoint

    if trainer.logger is not None:
        if trainer.weights_save_path != trainer.default_root_dir:
            # the user has changed weights_save_path, it overrides anything
            save_dir = trainer.weights_save_path
        else:
            save_dir = trainer.logger.save_dir or trainer.default_root_dir

        version = (
            trainer.logger.version
            if isinstance(trainer.logger.version, str)
            else f"version_{trainer.logger.version}"
        )
        ckpt_path = os.path.join(save_dir, str(
            trainer.logger.name), version, "checkpoints")
    else:
        ckpt_path = os.path.join(trainer.weights_save_path, "checkpoints")
    ckpt_path = trainer.training_type_plugin.broadcast(ckpt_path)

    return ckpt_path


def colorize_mask(mask):
    assert isinstance(mask, Image.Image)
    new_mask = mask.convert('P')
    new_mask.putpalette(palette)
    return new_mask


def crop(img, crop_bbox):
    """Crop from ``img``"""
    crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
    if img.dim() == 4:
        img = img[:, :, crop_y1:crop_y2, crop_x1:crop_x2]
    elif img.dim() == 3:
        img = img[:, crop_y1:crop_y2, crop_x1:crop_x2]
    elif img.dim() == 2:
        img = img[crop_y1:crop_y2, crop_x1:crop_x2]
    else:
        raise NotImplementedError(img.dim())
    return img
