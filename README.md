# Simple image classification engine

A basic image classification engine

The objective: [ML_assignment](ML_assignment.md)

# Environment

- Ubuntu 20.04
- Python 3.8
- Nvidia GeForce RTX 3070 Laptop GPU
- CUDA 11.6

# Installation
```bash
git clone https://github.com/Vanchez23/simple_img_classification_engine.git
cd simple_img_classification_engine
pip3 install -r requirements.txt
```

# How to prepare dataset

 - images - all images should be in one specific folder
 - annotation_file - pandas dataframe that containing the following columns:
 - - name - a name of an image
 - - class - integer representation of label
 - - label - string representation
 - - split - train, valid or test subset (optional)

[Sample dataset](data/sample_dataset)

# Configuration

[pipeline_config.yaml](pipeline_config.yaml)

> - **reinit_fit_model** - reinitialize model weights for training
> - **load_predict_model** - load a model for prediction from database (or from checkpoint)

Add a new dataset:
> - **dataloader.annotation_file** - path to annotation_file
> - **dataloader.img_dir** - path to images
> - **dataloader.num_classes** - number of classes in a dataset
> - **dataloader.split_dataset** - split dataset (If it wasn't done before)
> - **dataloader.batch_size** - number of samples in a batch
> - **dataloader.num_workers** - number of workers (To speedup batch collection, set **dataloader.num_workers** to the same value as **dataloader.batch_size**)

Customize training:
> - **checkpoint_cfg.experiment_path** - path to root experiment folder
> - **checkpoint_cfg.experiment_name** - experiment_name
> - **checkpoint_cfg.eval_metric** - a validation metric to create a checkpoint
> - **checkpoint_cfg.eval_criterion** - how to compare **checkpoint_cfg.eval_metric** ("less" or "more")
> - **checkpoint_cfg.save_each_epoch** - epoch save rate (i.e. if **checkpoint_cfg.save_each_epoch** is equal 2, then checkpoints will be saved every 2 epochs)

> - **trainer.max_epochs** - number of epochs
> - **trainer.gpus** - a list or a single value of gpus

Set [Wandb](https://wandb.ai) logger (Optional):
> - **trainer.wandb_logger.project** - a name of the project
> - **trainer.wandb_logger.name** - a name of the experiment

# Create database
```bash
python3 create_db.py
```

# Run
 1. [Prepare dataset](#how-to-prepare-dataset)
 2. [Set config](#configuration)
 3. [Create database](#create_database)
 4. Run:
```bash
python3 main.py
```