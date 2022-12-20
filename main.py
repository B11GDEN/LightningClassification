import timm
import wandb
from pathlib import Path

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import RichProgressBar
from pytorch_lightning.loggers import WandbLogger

from pl_bolts.datamodules import CIFAR10DataModule
from src.datamodule.imagenet import ImagenetDataModule

from src.utils.transform import get_cifar10_transform, get_imagenet_transform
from src.model.lit_classifier import LitClassifier
from src.model.components import LinearAttention, get_all_parent_layers
from timm.models.vision_transformer import VisionTransformer, Attention


def main():
    logger = WandbLogger(
        project='Transformer',
        config='configs/default.yaml'
    )

    # define datamodule
    match wandb.config['dataset_name']:
        case 'cifar10':
            train_transforms, test_transforms = get_cifar10_transform()
            datamodule = CIFAR10DataModule(
                data_dir='datasets',
                batch_size=wandb.config['batch_size'],
                num_workers=wandb.config['num_workers'],
                train_transforms=train_transforms,
                test_transforms=test_transforms,
                val_transforms=test_transforms,
            )
        case 'imagenet':
            train_transforms, test_transforms = get_imagenet_transform()
            datamodule = ImagenetDataModule(
                data_dir=Path('datasets/ILSVRC/DATA/CLS-LOC'),
                config_dir=Path('datasets'),
                batch_size=wandb.config['batch_size'],
                num_workers=wandb.config['num_workers'],
            )
        case _:
            print('Unknown dataset name')
            wandb.finish()
            return

    # define net and change attention type
    net = VisionTransformer()
    for parent_layer, last_token in get_all_parent_layers(net, Attention):
        setattr(
            parent_layer, last_token,
            LinearAttention(
                dim=768, num_heads=12,
                qkv_bias=True, kv_drop=0., proj_drop=0.,
                q_kernel='l2', k_kernel='l2',
            )
        )
    model = LitClassifier(net=net)

    # define callbacks
    callbacks = [
        LearningRateMonitor(logging_interval='epoch'),
        RichProgressBar()
    ]

    # start train cycle
    trainer = Trainer(
        max_epochs=wandb.config['epochs'],
        accelerator=wandb.config['accelerator'],
        logger=logger,
        callbacks=callbacks,
        track_grad_norm=2,
    )

    trainer.fit(model, datamodule)
    trainer.test(model, datamodule=datamodule)

    wandb.finish()


if __name__ == '__main__':
    main()
