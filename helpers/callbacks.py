import pytorch_lightning as pl
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.utilities.cli import CALLBACK_REGISTRY


@CALLBACK_REGISTRY
class ValEveryNSteps(pl.Callback):

    def __init__(self, every_n_steps: int):
        self.last_run = None
        self.every_n_steps = every_n_steps

    def on_batch_end(self, trainer, pl_module):
        # Prevent Running validation many times in gradient accumulation
        if trainer.global_step == self.last_run:
            return
        else:
            self.last_run = None
        if trainer.global_step % self.every_n_steps == 0 and trainer.global_step != 0:
            trainer.training = False
            stage = trainer.state.stage
            trainer.state.stage = RunningStage.VALIDATING
            trainer._run_evaluate()
            trainer.state.stage = stage
            trainer.training = True
            trainer.logger_connector._epoch_end_reached = False
            self.last_run = trainer.global_step
