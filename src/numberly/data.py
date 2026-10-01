from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def build_loaders(batch_size=64):
    training_set = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )

    validation_set = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )

    training_dataloader = DataLoader(training_set, batch_size=64, shuffle=True)
    validation_dataloader = DataLoader(validation_set, batch_size=64, shuffle=False)

    return training_dataloader, validation_dataloader
