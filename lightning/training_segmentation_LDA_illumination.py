import pytorch_lightning as pl
import torch
import torch.nn as nn
from OpticalModel import OpticalModel
from AcquisitionModel import AcquisitionModel
from SpectralFilterGenerator import SpectralFilterGeneratorLDA
import segmentation_models_pytorch as smp
from DataModule import VCSELDataModule
from pytorch_lightning.loggers import TensorBoardLogger
import torchmetrics
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from pytorch_lightning.profilers import PyTorchProfiler
from SegmentationModule import SegmentationModule
import datetime

datamodule = VCSELDataModule(data_dir="./dataset_images_VCSELs_only_beginning_but_not_full/train",
                                batch_size=8,
                                num_workers=3)


# segmentation_module = SegmentationModule_oneshot(log_images_every_n_steps=200,encoder_weights=None,log_dir=log_dir+'/'+ name)

early_stop_callback = EarlyStopping(
                            monitor='val_loss',  # Metric to monitor
                            patience=15,        # Number of epochs to wait for improvement
                            verbose=True,
                            mode='min'          # 'min' for metrics where lower is better, 'max' for vice versa
                            )

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',      # Metric to monitor
    dirpath='checkpoints/',  # Directory path for saving checkpoints
    filename='checkpoint_ref_2shots',  # Checkpoint file name
    save_top_k=1,            # Save the top k models
    mode='min',              # 'min' for metrics where lower is better, 'max' for vice versa
    save_last=True           # Additionally, save the last checkpoint to a file named 'last.ckpt'
)

name = "training_LDA_PCA_2shots"
log_dir = 'tb_logs_revival'

logger = TensorBoardLogger(log_dir, name=name)
torch.autograd.set_detect_anomaly(True)
profiler = PyTorchProfiler(
    on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir + f'/{name}/profiler0'),
    trace_memory=True, schedule=torch.profiler.schedule(skip_first=20,wait=1, warmup=1, active=20, repeat=1)
)
segmentation_module = SegmentationModule(log_images_every_n_steps=400,encoder_weights=None,log_dir=log_dir)


trainer = pl.Trainer(profiler=profiler,
                        logger=logger,
                        accelerator="gpu",
                        max_epochs=500,
                        log_every_n_steps=40,
                        callbacks=[early_stop_callback, checkpoint_callback])

trainer.fit(segmentation_module, datamodule)

segmentation_module.writer.close()