from logging import warning
from typing import Dict, Tuple


from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
try:
    from pytorch_lightning.loggers import WandbLogger
except:
    print('Wandb logger isn\'t installed. To install run: pip install wandb')
import albumentations as A
from albumentations.pytorch import ToTensorV2

from trainer import Classifier
from create_dataset import CustomDataset

class Pipeline:

    def __init__(self, cfg:Dict):

        self._cfg = cfg
        trainer_cfg = self._cfg['trainer']
        logger = WandbLogger(**trainer_cfg['wandb_logger'])
        self.trainer = Trainer(gpus = trainer_cfg['gpus'], 
                        deterministic= trainer_cfg['deterministic'],
                        logger=logger)
        self.classifier = Classifier(self._cfg['num_classes'])


    def _set_dataloaders(self) -> Tuple[DataLoader,DataLoader, DataLoader]:

        annotation_file = self._cfg['annotation_file']
        img_dir = self._cfg['img_dir']

        df = CustomDataset.read_annotation_file(annotation_file)

        necessary_columns = ('name','label')
        for column_name in necessary_columns:
            assert column_name in df.columns, f"Columns: {necessary_columns} should be in the dataframe"

        if self._cfg['split_dataset']:
            df_train, df_valid, df_test = CustomDataset.split_df(df, self._cfg['random_state'])
        else:
            assert 'split' in df.columns
            df_train = df[df['split'] == 'train']
            df_valid = df[df['split'] == 'valid']
            df_test = df[df['split'] == 'test']

        train_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), 
                        std=(0.229, 0.224, 0.225)),
            A.Resize(256,256),
            ToTensorV2()
        ])
        valid_transform = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), 
                        std=(0.229, 0.224, 0.225)),
            A.Resize(256,256),
            ToTensorV2()
        ])
        train_dataset = CustomDataset(df_train, img_dir, train_transform)
        valid_dataset = CustomDataset(df_valid, img_dir, valid_transform)
        test_dataset = CustomDataset(df_test, img_dir, valid_transform)

        train_loader = DataLoader(train_dataset, batch_size = 64, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size = 64, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size = 64, shuffle = False)

        return train_loader, valid_loader, test_loader



    def run(self):
        
        train_loader, valid_loader, test_loader = self._set_dataloaders()
        self.trainer.fit(self.classifier, train_dataloaders=train_loader,
                    val_dataloaders=valid_loader)
        result = self.trainer.test(self.classifier, dataloaders=test_loader)


        
