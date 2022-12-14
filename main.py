import timm
import wandb
from pathlib import Path

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import RichProgressBar
from pytorch_lightning.loggers import WandbLogger

from src.datamodule.imagenet import ImagenetDataModule
from src.datamodule.cifar100 import Cifar100DataModule
from src.datamodule.cifar10 import Cifar10DataModule
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
            datamodule = Cifar10DataModule(
                data_dir=Path('./datasets/cifar-10-batches-py'),
                batch_size=wandb.config['batch_size'],
                num_workers=wandb.config['num_workers'],
            )
        case 'cifar100':
            datamodule = Cifar100DataModule(
                data_dir=Path('./datasets/cifar-100-python'),
                batch_size=wandb.config['batch_size'],
                num_workers=wandb.config['num_workers'],
            )
        case 'imagenet':
            datamodule = ImagenetDataModule(
                data_dir=Path('./datasets/ILSVRC/Data/CLS-LOC'),
                config_dir=Path('datasets'),
                batch_size=wandb.config['batch_size'],
                num_workers=wandb.config['num_workers'],
            )
        case _:
            print('Unknown dataset name')
            # wandb.finish()
            return

    # define net and change attention type
    # net = timm.create_model(wandb.config['model_name'], pretrained=False)
    net = VisionTransformer(
        img_size=32,
        patch_size=8,
        num_classes=10,
        embed_dim=192,
        depth=12,
        num_heads=3,
    )
    for parent_layer, last_token in get_all_parent_layers(net, Attention):
        setattr(
            parent_layer, last_token,
            LinearAttention(
                dim=wandb.config['dim'],
                num_heads=wandb.config['num_heads'],
                qkv_bias=wandb.config['qkv_bias'],
                kv_drop=wandb.config['kv_drop'],
                proj_drop=wandb.config['proj_drop'],
                q_kernel=wandb.config['q_kernel'],
                k_kernel=wandb.config['k_kernel'],
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
        limit_train_batches=1.0,
    )

    trainer.fit(model, datamodule)
    trainer.test(model, datamodule=datamodule)

    # wandb.finish()


if __name__ == '__main__':
    main()
