import pytorch_lightning as pl
from helpers.cli import ConditioningLightningCLI

ConditioningLightningCLI(pl.LightningModule,
                         pl.LightningDataModule,
                         subclass_mode_model=True,
                         subclass_mode_data=True,
                         parser_kwargs={'error_handler': None})
