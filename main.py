import timm
import wandb

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import RichProgressBar
from pytorch_lightning.loggers import WandbLogger

from pl_bolts.datamodules import CIFAR10DataModule, ImagenetDataModule

from src.utils.transform import get_cifar10_transform, get_imagenet_transform
from src.model.lit_classifier import LitClassifier
from src.model.vit import VisionTransformer


def main():
    logger = WandbLogger(
        project='Transformer',
        config='configs/default.yaml'
    )

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
                data_dir='datasets/tiny-imagenet-200',
                batch_size=wandb.config['batch_size'],
                num_workers=wandb.config['num_workers'],
                # train_transforms=train_transforms,
                # test_transforms=test_transforms,
                # val_transforms=test_transforms,
            )
        case _:
            print('Unknown dataset name')
            wandb.finish()
            return

    model = LitClassifier(
        net= VisionTransformer(
            embed_dim=256,
            hidden_dim=512,
            num_heads=8,
            num_layers=6,
            patch_size=16,
            num_channels=3,
            num_patches=4,
            num_classes=datamodule.num_classes,
            dropout=0.0
        )
    )

    callbacks = [
        LearningRateMonitor(logging_interval='epoch'),
        RichProgressBar()
    ]

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
