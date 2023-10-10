"""
PyTorch Quickstart codes from https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html

This code is modified to use WandB for logging and tracking.
The standard output is also modified to be removed and replaced with WandB logging.
"""
from dataclasses import dataclass, asdict

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from tqdm import tqdm, trange

import wandb

from quickstart_nn import NeuralNetwork


@dataclass
class Config:
    batch_size: int = 64
    epochs: int = 10
    learning_rate: float = 1e-3
    log_interval: int = 100


config = Config()

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)


# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=config.batch_size)
test_dataloader = DataLoader(test_data, batch_size=config.batch_size)


# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


nn_model = NeuralNetwork().to(device)


ce_loss = nn.CrossEntropyLoss()
sgd_optim = torch.optim.SGD(nn_model.parameters(), lr=config.learning_rate)


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in tqdm(
        enumerate(dataloader), total=size / config.batch_size, desc="Batch"
    ):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        wandb.log({"loss": loss})
        if batch % config.log_interval == 0:
            loss, current = loss.item(), batch * len(X)

            with torch.inference_mode():
                eval(test_data, nn_model)


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    wandb.summary["test_accuracy"] = correct


def eval(data, model):
    classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    model.eval()
    x, y = data[0][0], data[0][1]
    with torch.no_grad():
        x = x.to(device)
        pred = model(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]


if __name__ == "__main__":
    wandb.init(
        project="pytorch-quickstart",
        config=asdict(config),
        name="ruby-plasma-2",
    )

    for t in trange(config.epochs, desc="Epoch"):
        train_loop(train_dataloader, nn_model, ce_loss, sgd_optim)
        test(test_dataloader, nn_model, ce_loss)
    print("Done!")

    torch.save(nn_model.state_dict(), "./model.pth")
    wandb.save("./model.pth")

    artifact = wandb.Artifact("fashion_mnist", type="model")
    artifact.add_file("./model.pth")
    wandb.log_artifact(artifact)

    eval(test_data, nn_model)

    wandb.finish()
