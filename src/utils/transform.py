import torchvision
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization, imagenet_normalization


def get_cifar10_transform():

    train_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(224),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            cifar10_normalization(),
        ]
    )

    test_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(224),
            torchvision.transforms.ToTensor(),
            cifar10_normalization(),
        ]
    )

    return train_transforms, test_transforms

def get_imagenet_transform():

    train_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            imagenet_normalization(),
        ]
    )

    test_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            imagenet_normalization(),
        ]
    )

    return train_transforms, test_transforms