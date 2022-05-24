import os.path as osp
import warnings
from typing import Dict
from copy import deepcopy

import pandas as pd
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
        self.classifier = Classifier(self._cfg)
        self.save_result_path = self.classifier.save_result_path
        
        trainer_cfg = self._cfg['trainer']
        if 'save_dir' not in trainer_cfg['wandb_logger']:
            trainer_cfg['wandb_logger']['save_dir'] = self.classifier.experiment_path

        logger = WandbLogger(**trainer_cfg['wandb_logger'])
        del trainer_cfg['wandb_logger']
        self.trainer = Trainer(logger, **trainer_cfg)

    def run(self) -> Dict[str, int]:
        
        self.trainer.fit(self.classifier)
        result = self.trainer.test(self.classifier)
        df = pd.DataFrame(result)
        df.to_csv(self.save_result_path, index=False)
        return result