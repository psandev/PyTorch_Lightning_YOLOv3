import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import GPUStatsMonitor, ModelCheckpoint
from utils.save_code_files import *
from models_focal import *
import logging
from utils.datasets_albumentations import ListDataset as List_alb
from utils.datasets_albumentations import get_transform
from test import *
from omegaconf import OmegaConf
from loguru import logger
from wcmatch.pathlib import Path
from terminaltables import AsciiTable
from datetime import datetime
import argparse
import wandb
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_ranger import Ranger


class COCODataModule(pl.LightningDataModule):
    def __init__(self, train_ds, val_ds, conf: OmegaConf):
        super().__init__()
        self.batch_size = conf.run.batch_size
        self.train_ds = train_ds
        self.val_ds = val_ds

    def train_dataloader(self):
        return DataLoader(self.train_ds,
                          batch_size=conf.run.batch_size,
                          shuffle=True,
                          num_workers=conf.run.n_cpu,
                          pin_memory=True,
                          collate_fn=self.train_ds.collate_fn,
                          )

    def val_dataloader(self):
        return DataLoader(self.val_ds,
                          batch_size=conf.run.batch_size,
                          shuffle=False,
                          num_workers=conf.run.n_cpu,
                          pin_memory=True,
                          collate_fn=self.val_ds.collate_fn,
                          )


class NET(pl.LightningModule):
    def __init__(self, model, optimizer,  detect):
        super().__init__()
        self.model = model
        self.optim = optimizer
        self.detect = detect

    def forward(self, x, targets):
        return self.model(x, targets)

    def configure_optimizers(self):
        return Ranger(self.model.parameters(), lr=1e-4, weight_decay=1e-5, eps=1e-9)

    def training_step(self, batch, batch_idx):
        imgs, targets = batch[1:]
        imgs = Variable(imgs.to(device))
        targets = Variable(targets.to(device), requires_grad=False)
        loss, outputs = self.model(imgs, targets)
        self.log('train_loss',loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True
                 )
        return loss

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        items.pop("loss", None)
        return items

    def validation_step(self, batch, batch_idx):
        imgs, targets = batch[1:]
        loss, outputs = self.model(imgs, targets)
        self.log('val_loss',loss,
                  on_step=True,
                  on_epoch=True,
                  prog_bar=True,
                  logger=True
                  )
        return loss

    def validation_epoch_end(self, loss):
        if self.detect:
            metrics_output = evaluate(
                self.model,
                path=conf.data.val,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=conf.run.img_size,
                batch_size=8,
                cache_path=conf.data.cache_path,
                cache=conf.data.cache
            )

            if metrics_output is not None:
                precision, recall, AP, f1, ap_class = metrics_output
                self.log('precision', precision.mean(), on_epoch=True, prog_bar=True, logger=True)
                self.log('recal', recall.mean(), on_epoch=True, prog_bar=True, logger=True)
                self.log('f1', f1.mean(), on_epoch=True, prog_bar=True, logger=True)
                self.log('mAP', AP.mean(), on_epoch=True, prog_bar=True, logger=True)

                ap_table = [["Index", "Class name", "AP"]]
                for i, c in enumerate(ap_class):
                    ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
                print(AsciiTable(ap_table).table)
                print(f"---- mAP {AP.mean()}")
            else:
                print("---- mAP not measured (no detections found by model)")

def objective(trial: optuna.trial.Trial) -> float:
    optimizer_name = trial.suggest_categorical('optimizer', ['Ranger', 'AdamW'])
    lr_a = trial.suggest_loguniform('lr_a', 1e-5, 1e-1)
    weight_decay_a = trial.suggest_loguniform('wd_a', 1e-5, 1e-1)
    eps_a = trial.suggest_loguniform('eps_a', 1e-10, 1e-8)
    lr_r = trial.suggest_loguniform('lr_r', 1e-5, 1e-1)
    weight_decay_r = trial.suggest_loguniform('wd_r', 1e-5, 1e-1)
    eps_r = trial.suggest_loguniform('eps_r', 1e-10, 1e-8)
    darknet = Darknet(conf.run.model_def, apply_focal_loss=conf.run.apply_focal_loss)
    dataModule = COCODataModule(train_ds, val_ds, conf)
    if optimizer_name == 'AdamW':
        optimizer = optim.AdamW(darknet.parameters(), lr=lr_a, weight_decay=weight_decay_a, eps=eps_a)
    if optimizer_name == 'Ranger':
        optimizer = Ranger(darknet.parameters(), lr=lr_r, weight_decay=weight_decay_r, eps=eps_r)
    model = NET(darknet, optimizer, detect=conf.run.detect)
    trainer = pl.Trainer(
        logger=True,
        limit_train_batches=0.3,
        limit_val_batches=0.2,
        checkpoint_callback=False,
        max_epochs=5,
        gpus=-1 if torch.cuda.is_available() else None,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_acc")],
    )
    hyperparameters = dict(optim=optimizer_name,
                           lr_a=lr_a,
                           wd_a = weight_decay_a,
                           eps_a = eps_a,
                           lr_r=lr_r,
                           wd_r=weight_decay_r,
                           eps_r=eps_r
                           )
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model, datamodule=dataModule)

    return trainer.callback_metrics["val_loss_epoch"].item()


if __name__ == "__main__":
    level = 'DEBUG'
    darknet_pretrained = False
    experiment_root = 'experiments'
    exp_root_path = Path(experiment_root)
    wandb.login()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="config/config.yaml", help="Experiment configuration file")
    # parser.add_argument("--restore_from_dir", type=str, default='experiments/train/exp_26-Mar-2021_19_42_47)',
    #                     help="Start from checkpoint")
    parser.add_argument("--restore_from_dir", type=str, default=None,
                        help="Start from checkpoint")
    opt = parser.parse_args()
    conf = OmegaConf.create({'run': {'config_file': opt.config_file}})
    conf.run.restore_from_dir = opt.restore_from_dir
    experiment_dir = exp_root_path / 'train' / f'exp_{datetime.now().strftime("%d-%b-%Y_%H_%M_%S")}'
    conf.run.experiment_dir = experiment_dir.as_posix()
    conf = OmegaConf.merge(conf, OmegaConf.load(conf.run.config_file))
    # print(OmegaConf.to_yaml(conf))

    log_file = experiment_dir / f'log_{datetime.now().strftime("%d-%b-%Y_%H_%M_%S)")}.log'

    logger.opt(record=True).add(
                                log_file,
                                format=' {time:YYYY-MMM HH:mm:ss} {name}:{function}:{line} <lvl>{message}</>',
                                level='DEBUG', rotation='5 MB'
                                 )

    logger.opt(colors=True).info(OmegaConf.to_yaml(conf))
    experiment_dir.mkdir(exist_ok=True)

    save_code_files(experiment_dir,os.path.dirname(__file__))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_names = load_classes(conf.data.names)
    train_ds = ListDataset(
                           image_list_file=conf.data.train,
                           cache_path= conf.data.cache_path,
                           multiscale=conf.run.multiscale_training,
                           img_size=conf.run.img_size,
                           transform=AUGMENTATION_TRANSFORMS,
                           cache=conf.data.cache
                           )

    val_ds = ListDataset(
                         image_list_file=conf.data.val,
                         cache_path=conf.data.cache_path,
                         multiscale=conf.run.multiscale_training,
                         img_size=conf.run.img_size,
                         transform=AUGMENTATION_TRANSFORMS,
                         cache=conf.data.cache
                         )


    # train_transform, test_transform = get_transform(img_size=conf.run.img_size,
    #                                                 darknet_pretrained=darknet_pretrained)
    # train_ds = List_alb(conf.data.train, train_transform,img_size=conf.run.img_size)
    #
    # val_ds = List_alb(conf.data.val, test_transform, img_size=conf.run.img_size)

    darknet = Darknet(conf.run.model_def, apply_focal_loss=conf.run.apply_focal_loss)
    # darknet = Darknet(conf.run.model_def)
    optimizer = optim.Adam(darknet.parameters())
    model = NET(darknet, optimizer, detect=conf.run.detect)
    model = model.to(device)
    if not conf.run.restore_from_dir:
        logger.opt(colors=True).info('Loading pretrained weights')
        model.apply(weights_init_normal)
        darknet_pretrained = True
    dataModule = COCODataModule(train_ds, val_ds, conf)
    wandb_logger = WandbLogger(project="PL-YOLOv3",
                               save_dir=experiment_dir,
                               offline=False,
                               name='focal'
                               )
    tsrb_logger = TensorBoardLogger(name="PL-YOLOv3", save_dir=experiment_dir)

    checkpoint_callback = ModelCheckpoint(dirpath=experiment_dir, mode='min', monitor='val_loss')
    if not conf.run.restore_from_dir:
        if not conf.run.hp_optimize:
            logger.opt(colors=True).info('Training from scratch')
            trainer = pl.Trainer(
                default_root_dir=experiment_dir,
                # logger=wandb_logger,
                max_epochs=conf.run.epochs,
                gpus=-1,
                accumulate_grad_batches=1,
                gradient_clip_val=0.5,
                benchmark=True,
                # callbacks=[GPUStatsMonitor(), checkpoint_callback],
                auto_lr_find=True,
                # limit_train_batches=0.02,
                # limit_val_batches=0.1,
                num_sanity_val_steps=0,
                # fast_dev_run=100,
                auto_scale_batch_size = 'binsearch',
                # val_check_interval=1
            )
            trainer.fit(model, dataModule)
        else:
            optuna.logging.enable_propagation()
            logger.opt(colors=True).info('Start Optimization')
            study_name = 'first_try'
            logger.opt(colors=True).info('Start Optimization')
            df_name = Path(conf.run.experiment_dir)/ 'optuna_history.csv'
            study = optuna.create_study(study_name=study_name,
                                        direction='minimize',
                                        pruner=optuna.pruners.MedianPruner(),
                                        # storage=f'sqlite://{storage}',
                                        load_if_exists=True)
            study.optimize(objective, n_trials=20)

            logger.opt(colors=True).info(f'Number of Trial: {len(study.trials)}')
            logger.opt(colors=True).info(f'Best Trial:')
            trial = study.best_trial
            logger.opt(colors=True).info(f'Best Value: {trial.value}')
            logger.opt(colors=True).info(f'Best Params: {study.best_params}')
            df = study.trials_dataframe(attrs={'number', 'value', 'params', 'state'})
            df.to_csv(df_name, index=False)
    else:
        logger.opt(colors=True).info('Loading checkpoint',)
        ckpt_file = list(Path(conf.run.restore_from_dir).glob('*.ckpt'))[0].as_posix()
        trainer = pl.Trainer(resume_from_checkpoint=ckpt_file,
                             # logger=wandb_logger,
                             gpus=-1,
                             auto_scale_batch_size='binsearch',
                             num_sanity_val_steps=0,
                             callbacks=[GPUStatsMonitor(), checkpoint_callback],
                             benchmark=True,
                             )

        trainer.fit(model, dataModule)

