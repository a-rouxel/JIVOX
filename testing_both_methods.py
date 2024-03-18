import numpy as np
import pytorch_lightning as pl
from lightning.data_modules.Images_segmentation_datamodule import VCSELDataModule

from lightning.lightning_modules.Segmentation_sinus_shape_lightning_mod import SegmentationModuleFullNN
from lightning.lightning_modules.Segmentation_arb_shape_lightning_mod import SegmentationModule



data_dir = "./dataset_images/test/"

readout_noise_level_np = np.linspace(0.001, 0.3, 30)

datamodule = VCSELDataModule(data_dir=data_dir,
                             batch_size=8,
                             num_workers=3)

trainer = pl.Trainer(accelerator="gpu",
                     max_epochs=500,
                    log_every_n_steps=40)

for readout_noise_level in readout_noise_level_np:

        # TESTING SINUSOIDE + SEGMENTATION
        segmentation_module = SegmentationModuleFullNN(log_images_every_n_steps=500,
                                                 encoder_weights=None,
                                                 unet_checkpoint="./checkpoints/sinusoide_and_seg.ckpt",
                                                 sinus_checkpoint="./checkpoints/sinusoide_and_seg.ckpt",
                                                 readout_noise_level=readout_noise_level,
                                                 )
        trainer.test(segmentation_module,datamodule)

        # TESTING ARBITRARY SHAPE + SEGMENTATION
        segmentation_module = SegmentationModule(log_images_every_n_steps=400,
                                                 encoder_weights=None,
                                                 unet_checkpoint="./checkpoints/seg.ckpt",
                                                 readout_noise_level=readout_noise_level,)
        trainer.test(segmentation_module, datamodule)

