import pytorch_lightning as pl
import torch
from lightning.data_modules.Sinus_illumination_datamodule import SpectralIlluminationModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.profilers import PyTorchProfiler
from lightning.lightning_modules.SpectralIllumination_lightning_mod  import ResNet1DLightning_optuna
from physics_models.OpticalModel import OpticalModel


data_dir = "./dataset_images/train"
name = "training_sinusoide"
log_dir = 'tb_logs'


params = {'batch_size': 32,
                     'learning_rate': 0.00038,
                     'num_blocks_layer0': 3,
                     'channels_layer0': 32,
                     'num_blocks_layer1': 2,
                     'channels_layer1': 64,
                     'num_blocks_layer2': 4,
                     'channels_layer2': 128,
                     'num_blocks_layer3': 2,
                     'channels_layer3': 128,
                     'common_dense_size': 256}

datamodule = SpectralIlluminationModule(data_dir=data_dir,
                                batch_size=params["batch_size"],
                                num_workers=3)


# segmentation_module = SegmentationModule_oneshot(log_images_every_n_steps=200,encoder_weights=None,log_dir=log_dir+'/'+ name)

early_stop_callback = EarlyStopping(
                            monitor='val_loss',  # Metric to monitor
                            patience=15,        # Number of epochs to wait for improvement
                            verbose=True,
                            mode='min'          # 'min' for metrics where lower is better, 'max' for vice versa
                            )


logger = TensorBoardLogger(log_dir, name=name)

profiler = PyTorchProfiler(
    on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir + f'/{name}/profiler0'),
    trace_memory=True, schedule=torch.profiler.schedule(skip_first=20,wait=1, warmup=1, active=20, repeat=1)
)


optical_model = OpticalModel()
torch.autograd.set_detect_anomaly(True)

# Now load the state dict into your model
segmentation_module = ResNet1DLightning_optuna(optical_model,
                                                    learning_rate=params["learning_rate"],
                                                   num_blocks_per_layer=[params[f"num_blocks_layer{i}"] for i in range(4)],
                                                   channels_per_block=[params[f"channels_layer{i}"] for i in range(4)],
                                                   common_dense_size=params["common_dense_size"])

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',      # Metric to monitor
    dirpath=log_dir + f'/{name}/checkpoints/',  # Directory path for saving checkpoints
    filename='sinusoide_only',  # Checkpoint file name
    save_top_k=1,            # Save the top k models
    mode='min',              # 'min' for metrics where lower is better, 'max' for vice versa
    save_last=True           # Additionally, save the last checkpoint to a file named 'last.ckpt'
)

# Your existing setup code for TensorBoardLogger, PyTorchProfiler, and model initialization...
trainer = pl.Trainer(
    profiler=profiler,
    logger=logger,
    accelerator="gpu",
    max_epochs=2,
    log_every_n_steps=200,
    callbacks=[early_stop_callback, checkpoint_callback]  # Add checkpoint_callback here
)

trainer.fit(segmentation_module, datamodule)
segmentation_module.writer.close()