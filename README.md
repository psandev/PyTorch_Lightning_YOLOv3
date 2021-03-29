This is a YOLOv3 implemented in Pytorch.
The model parsing and original loss were taken from this excelet repo [Pytorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3).

## My contributions:
* Focal loss
* Caching dataset
* [Albumetations](https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/) transforms
* Hyper parameters optimization with [Optuna](https://optuna.readthedocs.io/en/stable/)
* [Pytorch-Lightning](https://pytorch-lightning.readthedocs.io/en/stable/starter/new-project.html) for rapid prototyping:
    * LR Finder.
    * Validation check before training.
    * Control of train and validation number of batches.
    * Separation betwen data and coode: DataModule and PyTorchLightningModule.
    * Automatic metrics tracking and logging into many supported logging frameworks.
* Improved project structure with code saving, hyperparameters and various other outputs logging.
    * Centralized parameters handling with [OmegaConf](https://omegaconf.readthedocs.io/en/2.0_branch/) - YAML based config manager.
    * [WandB](https://docs.wandb.ai/), Tensorboard and [Loguru](https://loguru.readthedocs.io/en/stable/) loggers.
       
