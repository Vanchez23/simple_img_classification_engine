import warnings
from typing import Tuple, Dict, List
from copy import deepcopy
import os
import os.path as osp
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
import pytorch_lightning as pl
import albumentations as A
from albumentations.pytorch import ToTensorV2

from .dataset import CustomDataset

class Classifier(pl.LightningModule):
    
    def __init__(self, cfg: Dict):
        super().__init__()
        self._cfg = deepcopy(cfg)
        
        self.num_classes = self._cfg['dataloader']['num_classes']
        self.img_dir = self._cfg['dataloader']['img_dir']

        self.experiment_path = self._cfg['checkpoint_cfg']['experiment_path']
        if not self.experiment_path:
            self.experiment_path = 'experiments'
        self.created_at = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.experiment_path = osp.join(self.experiment_path, self.created_at+self._cfg['checkpoint_cfg']['experiment_name'])
        self.save_checkpoint_path = osp.join(self.experiment_path, "weights")
        self.save_result_path = osp.join(self.experiment_path, "metrics.csv")
        os.makedirs(self.save_checkpoint_path, exist_ok=True)

        self.eval_metric = self._cfg['checkpoint_cfg']['eval_metric']
        self.eval_criterion = self._cfg['checkpoint_cfg']['eval_criterion']        
        if self.eval_metric is None:
            self.eval_metric = 'loss'
            self.eval_criterion = 'less'
        if self.eval_criterion == 'less':
            self.best_metric_value = float('inf')
        elif self.eval_criterion == 'more':
            self.best_metric_value = 0
        else:
            warnings.warn(f'Evaluation metric is "{self.eval_metric}", but evaluation criterion should be "less" or "more", got: {self.eval_criterion}')


        self.model = models.resnet50(pretrained="imagenet")
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, self.num_classes, bias=True)
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.save_hyperparameters()
        self.configure_optimizers()

        self.save_each_epoch = self._cfg['checkpoint_cfg']['save_each_epoch']
        self.load_checkpoint_path = self._cfg['checkpoint_cfg']['load_path']
        self.load_checkpoint(self.load_checkpoint_path)

        self.create_datasets(self._cfg['dataloader']['annotation_file'],
                            self._cfg['dataloader']['split_dataset'], 
                            self._cfg['dataloader']['random_state'])



################# Dataloaders ###########################

    def create_datasets(self, annotation_file, split_dataset, random_state) -> None:
        (self.df_train, 
        self.df_valid, 
        self.df_test,
        self.labels) = self._split_df(annotation_file,
                                        split_dataset, 
                                        random_state)

        self.train_dataset = self.set_train_dataset(self.df_train)
        self.val_dataset = self.set_test_dataset(self.df_valid)
        self.test_dataset = self.set_test_dataset(self.df_test)

    def set_train_dataset(self, df: pd.DataFrame) -> CustomDataset:
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), 
                        std=(0.229, 0.224, 0.225)),
            A.Resize(256,256),
            ToTensorV2()
        ])
        return CustomDataset(df, self.img_dir, transform)

    def set_test_dataset(self, df: pd.DataFrame) -> CustomDataset:
        transform = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), 
                        std=(0.229, 0.224, 0.225)),
            A.Resize(256,256),
            ToTensorV2()
        ])
        return CustomDataset(df, self.img_dir, transform)


    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size = self._cfg['dataloader']['batch_size'], 
                                    shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size = self._cfg['dataloader']['batch_size'], 
                                    shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size = self._cfg['dataloader']['batch_size'], 
                                    shuffle=False)

################ Steps ############################

    def training_step(self, batch:List[torch.Tensor], batch_idx:int) -> Dict[str, float]:
        x,y = batch
        pred = self.model(x)
        loss = self.loss_func(pred, y)
        self.log('train/batch_loss', loss)

        pred = torch.argmax(pred, dim=1)
        metrics = self.compute_metrics(pred, y)
        for metric_name, value in metrics.items():
            self.log('train/batch_'+metric_name, value)
        
        return {'loss': loss, **metrics}

    def validation_step(self, batch:List[torch.Tensor], batch_idx:int) -> Dict[str, float]:
        x,y = batch
        pred = self.model(x)
        loss = self.loss_func(pred, y)
        self.log('valid/batch_loss', loss)

        pred = torch.argmax(pred, dim=1)
        metrics = self.compute_metrics(pred, y)
        for metric_name, value in metrics.items():
            self.log('valid/batch_'+metric_name, value)

        return {'loss': loss, **metrics}

    def test_step(self, batch:List[torch.Tensor], batch_idx:int) -> Dict[str, float]:
        x,y = batch
        pred = self.model(x)
        loss = self.loss_func(pred, y)
        self.log('test/batch_loss', loss)

        pred = torch.argmax(pred, dim=1)
        metrics = self.compute_metrics(pred, y)
        for metric_name, value in metrics.items():
            self.log('test/batch_'+metric_name, value)

        return {'loss': loss, **metrics}


    def predict(self, img:np.ndarray) -> str:
        transform = self.test_dataset.transform
        tensor = self.test_dataset.prepare_img(img, transform)
        tensor.unsqueeze_(0)
        predict = self.model(tensor)
        predict = torch.argmax(predict, dim=1)
        return self.labels[predict]

############### Epoch ends ################
    
    def _common_epoch_end(self, outputs:List[Dict[str,float]], split:str) -> Dict[str, float]:
        means = {split+'/epoch_'+name: 0 for name in outputs[0].keys()}
        
        for output in outputs:
            for name, value in output.items():
                means[split+'/epoch_'+name] += value/len(outputs)

        for name, value in means.items():
            self.log(name, value)

        return means

    def training_epoch_end(self, outputs:List[Dict[str,float]]) -> Dict[str, float]:
        self._common_epoch_end(outputs, 'train')

    def validation_epoch_end(self, outputs:List[Dict[str,float]]) -> Dict[str, float]:
        
        means = self._common_epoch_end(outputs, 'valid')

        if isinstance(self.save_each_epoch,int) and self.current_epoch % self.save_each_epoch == 0:
            self.save_checkpoint(means['valid/epoch_loss'], f"epoch{self.current_epoch}.pth", 
                                means['valid/epoch_'+self.eval_metric])
        if self.eval_criterion == 'less':
            if means['valid/epoch_'+self.eval_metric] < self.best_metric_value:
                self.save_checkpoint(means['valid/epoch_loss'], "best.pth",
                                    means['valid/epoch_'+self.eval_metric])
        elif self.eval_criterion == 'more':
            if means['valid/epoch_'+self.eval_metric] > self.best_metric_value:
                self.save_checkpoint(means['valid/epoch_loss'], "best.pth",
                                    means['valid/epoch_'+self.eval_metric])

        self.save_checkpoint(means['valid/epoch_loss'],
                            metric=means['valid/epoch_'+self.eval_metric])

    def test_epoch_end(self, outputs:List[Dict[str,float]]) -> Dict[str, float]:
        self._common_epoch_end(outputs, 'test')

############### Other #####################

    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer],List[torch.optim.lr_scheduler._LRScheduler]]:
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1)
        return [self.optimizer],[self.lr_scheduler]


    def _split_df(self, annotation_file:str, split_dataset:bool, random_state:int) -> Tuple[DataLoader,DataLoader, DataLoader]:

        df = CustomDataset.read_annotation_file(annotation_file)
        labels = df.drop_duplicates('class').sort_values('class')['label'].tolist()

        necessary_columns = ('name','class','label')
        for column_name in necessary_columns:
            assert column_name in df.columns, f"Columns: {necessary_columns} should be in the dataframe"

        if split_dataset:
            df_train, df_valid, df_test = CustomDataset.split_df(df, random_state)
        else:
            assert 'split' in df.columns
            df_train = df[df['split'] == 'train']
            df_valid = df[df['split'] == 'valid']
            df_test = df[df['split'] == 'test']

        ###################################
        warnings.warn("Dataframes are cutted for debugging. Please remove this code after debugging.")
        df_train = df_train.sample(frac=0.2, random_state=random_state)
        df_valid = df_valid.sample(frac=0.2, random_state=random_state)
        df_test = df_test.sample(frac=0.2, random_state=random_state)
        ###################################
        return df_train, df_valid, df_test, labels

    def save_checkpoint(self, loss:float, name:str='last.pth', metric=None) -> None:
        torch.save({
            'epoch': self.current_epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'loss': loss,
            'metric': metric,
            'metric_name': self.eval_metric,
            'best_metric_value': self.best_metric_value,
            'labels': self.labels,
            }, osp.join(self.save_checkpoint_path,name))

    def load_checkpoint(self, checkpoint_path:str) -> None:

        if checkpoint_path is None:
            return -1, -1, -1
        if not osp.exists(checkpoint_path):
            warnings.warn(f'Checkpoint path "{checkpoint_path}" doesn\'t exist.')
            return -1, -1, -1
        checkpoint = torch.load(checkpoint_path)
        # self.current_epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        self.best_metric_value = checkpoint['best_metric_value']
        metric = checkpoint['metric']
        metric_name = checkpoint['metric_name']
        loss = checkpoint['loss']
        self.labels = checkpoint['labels']

        return loss, metric, metric_name
    
    def compute_metrics(self, pred:torch.Tensor, y:torch.Tensor) -> Dict[str, float]:
        accuracy = torch.sum(y == pred).item() / (len(y) * 1.0)
        return {'accuracy': accuracy}