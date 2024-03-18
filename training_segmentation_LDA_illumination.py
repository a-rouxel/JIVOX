import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.profilers import PyTorchProfiler
import torch

from lightning.data_modules.Images_segmentation_datamodule import VCSELDataModule
from lightning.lightning_modules.Segmentation_arb_shape_lightning_mod import SegmentationModule


name = "training_seg_LDA"
log_dir = 'tb_logs'
data_dir_path = "./dataset/train"


datamodule = VCSELDataModule(data_dir=data_dir_path,
                                batch_size=8,
                                num_workers=3)

early_stop_callback = EarlyStopping(
                            monitor='val_loss',  # Metric to monitor
                            patience=15,        # Number of epochs to wait for improvement
                            verbose=True,
                            mode='min'          # 'min' for metrics where lower is better, 'max' for vice versa
                            )

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',      # Metric to monitor
    dirpath='checkpoints/',  # Directory path for saving checkpoints
    filename=name,  # Checkpoint file name
    save_top_k=1,            # Save the top k models
    mode='min',              # 'min' for metrics where lower is better, 'max' for vice versa
    save_last=True           # Additionally, save the last checkpoint to a file named 'last.ckpt'
)

logger = TensorBoardLogger(log_dir, name=name)

profiler = PyTorchProfiler(
    on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir + f'/{name}/profiler0'),
    trace_memory=True, schedule=torch.profiler.schedule(skip_first=20,wait=1, warmup=1, active=20, repeat=1)
)

segmentation_module = SegmentationModule(log_images_every_n_steps=400,encoder_weights=None,log_dir=log_dir)


trainer = pl.Trainer(profiler=profiler,
                        logger=logger,
                        accelerator="cpu",
                        max_epochs=500,
                        log_every_n_steps=40,
                        callbacks=[early_stop_callback, checkpoint_callback])

trainer.fit(segmentation_module, datamodule)

segmentation_module.writer.close()