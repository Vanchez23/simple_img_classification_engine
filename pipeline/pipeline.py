import os.path as osp
import yaml
from typing import Dict
from copy import deepcopy

from loguru import logger
import pandas as pd
from pytorch_lightning import Trainer
try:
    from pytorch_lightning.loggers import WandbLogger
    import wandb
except:
    logger.warning('Wandb logger isn\'t installed. To install run: pip install wandb')

from .trainer import Classifier
from .utils import load_img

class Pipeline(object):

    def __init__(self, cfg:Dict):

        self._cfg = deepcopy(cfg)
        self.classifier = Classifier(self._cfg)
        self.save_result_path = self.classifier.save_result_path
        
        self.trainer_cfg = self._cfg['trainer']
        if 'wandb_logger' in self.trainer_cfg and 'save_dir' not in self.trainer_cfg['wandb_logger']:
            self.trainer_cfg['wandb_logger']['save_dir'] = self.classifier.experiment_path
            logger.info(f"wandb save_dir is {self.classifier.experiment_path}")

        with open(osp.join(self.classifier.experiment_path,'pipeline_config.yaml'), 'w') as f:
            yaml.dump(self._cfg, f)
        
    def run(self) -> Dict[str, int]:

        if 'wandb_logger' in self.trainer_cfg:
            self.wandb_logger = WandbLogger(**self.trainer_cfg['wandb_logger'])
            del self.trainer_cfg['wandb_logger']
        else:
            self.wandb_logger = True        

        self.trainer = Trainer(logger=self.wandb_logger, **self.trainer_cfg)
        if self._cfg['reinit_fit_model']:
            self.classifier.reset_model()
            logger.info(f"The model has been reset")
        else:
            logger.info(f"Resume training from {self.classifier.load_checkpoint_path}")

        self.trainer.tune(self.classifier)
        self.trainer.fit(self.classifier)
        result = self.trainer.test(self.classifier)
        df = pd.DataFrame(result)
        df.to_csv(self.save_result_path, index=False)
        return result

    def predict(self, img_link:str) -> str:

        return self.classifier.predict(load_img(img_link)) 