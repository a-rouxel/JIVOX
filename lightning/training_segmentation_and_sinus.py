import pytorch_lightning as pl
import torch
from lightning.data_modules.Images_segmentation_datamodule import VCSELDataModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.profilers import PyTorchProfiler
from lightning.lightning_modules.Segmentation_sinus_shape_lightning_mod import SegmentationModuleFullNN


datamodule = VCSELDataModule(data_dir="./dataset_images_VCSELs_only_beginning_but_not_full/train",
                                batch_size=7,
                                num_workers=3)


# segmentation_module = SegmentationModule_oneshot(log_images_every_n_steps=200,encoder_weights=None,log_dir=log_dir+'/'+ name)

early_stop_callback = EarlyStopping(
                            monitor='val_loss',  # Metric to monitor
                            patience=30,        # Number of epochs to wait for improvement
                            verbose=True,
                            mode='min'          # 'min' for metrics where lower is better, 'max' for vice versa
                            )

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',      # Metric to monitor
    dirpath='checkpoints/',  # Directory path for saving checkpoints
    filename='sinusoide_dephased_full',  # Checkpoint file name
    save_top_k=1,            # Save the top k models
    mode='min',              # 'min' for metrics where lower is better, 'max' for vice versa
    save_last=True           # Additionally, save the last checkpoint to a file named 'last.ckpt'
)

name = "sinusoide_dephased_full"
log_dir = 'tb_logs_revival'

logger = TensorBoardLogger(log_dir, name=name)
torch.autograd.set_detect_anomaly(True)
profiler = PyTorchProfiler(
    on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir + f'/{name}/profiler0'),
    trace_memory=True, schedule=torch.profiler.schedule(skip_first=20,wait=1, warmup=1, active=20, repeat=1)
)

segmentation_module = SegmentationModuleFullNN(log_images_every_n_steps=500,
                                         encoder_weights=None,
                                         log_dir=log_dir+'/'+ name,
                                         unet_checkpoint="./checkpoints/checkpoint_ref_2shots.ckpt",
                                         sinus_checkpoint="./checkpoints/sinusoide_dephased-v8.ckpt",
                                         learning_rate=2e-5,
                                          weight_decay=1e-6
                                         )




trainer = pl.Trainer(profiler=profiler,
                        logger=logger,
                        accelerator="gpu",
                        max_epochs=500,
                        log_every_n_steps=100,
                        callbacks=[early_stop_callback, checkpoint_callback] )

trainer.fit(segmentation_module, datamodule)
segmentation_module.writer.close()
