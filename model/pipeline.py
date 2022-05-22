from logging import warning
from typing import Dict, Tuple
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer

from trainer import Classifier
try:
    from pytorch_lightning.loggers import WandbLogger
except:
    print('Wandb logger isn\'t installed. To install run: pip install wandb')

from create_dataset import CustomDataset

class Pipeline:

    def __init__(self, cfg:Dict):

        self._cfg = cfg
        trainer_cfg = self._cfg['trainer']
        logger = WandbLogger(**trainer_cfg['wandb_logger'])
        self.trainer = Trainer(gpus = trainer_cfg['gpus'], 
                        deterministic= trainer_cfg['deterministic'],
                        logger=logger,
                        log_gpu_memory=trainer_cfg['log_gpu_memory'])
        self.classifier = Classifier()


    def _set_dataloaders(self) -> Tuple[DataLoader,DataLoader, DataLoader]:

        annotation_file = self._cfg['annotation_file']
        img_dir = self._cfg['img_dir']

        df = CustomDataset.read_annotation_file(annotation_file)

        assert ['name','label'] in df.columns

        if self._cfg['split_dataset']:
            df_train, df_valid, df_test = CustomDataset.split_df(df, self._cfg['random_state'])
        else:
            assert 'split' in df.columns
            df_train = df[df['split'] == 'train']
            df_valid = df[df['split'] == 'valid']
            df_test = df[df['split'] == 'test']

        train_dataset = CustomDataset(df_train, img_dir)
        valid_dataset = CustomDataset(df_valid, img_dir)
        test_dataset = CustomDataset(df_test, img_dir)

        train_loader = DataLoader(train_dataset, batch_size = 64, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size = 64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size = 64, shuffle = False)

        return train_loader, valid_loader, test_loader



    def run(self):
        
        train_loader, valid_loader, test_loader = self._set_dataloaders()
        self.trainer.fit(self.classifier, train_dataloaders=train_loader,
                    val_dataloaders=valid_loader)
        result = self.trainer.test(self.classifier, dataloaders=test_loader)


        
