import warnings
from typing import Dict
from copy import deepcopy

from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
try:
    from pytorch_lightning.loggers import WandbLogger
    import wandb
except:
    warnings.warn('Wandb logger isn\'t installed. To install run: pip install wandb')

from .trainer import Classifier

class Pipeline(object):

    def __init__(self, cfg:Dict):

        self._cfg = deepcopy(cfg)
        trainer_cfg = self._cfg['trainer']
        logger = WandbLogger(**trainer_cfg['wandb_logger'])
        self.trainer = Trainer(gpus = trainer_cfg['gpus'], 
                        deterministic= trainer_cfg['deterministic'],
                        logger=logger)
        self.classifier = Classifier(self._cfg)

    def run(self) -> None:
        
        self.trainer.fit(self.classifier)
        result = self.trainer.test(self.classifier)


        
