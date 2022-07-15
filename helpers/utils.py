import os

import pytorch_lightning as pl


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
