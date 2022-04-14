import os.path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()  # flatten the 2D input from (1,28,28) to (784)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),  # this is basically: torch.mm(x, W) + b , W=(784,512), b=(1,512)
            nn.ReLU(),  # this is an activation function
            nn.Linear(512, 512),  # Note: the linear layer does not come with an activation
            nn.ReLU(),  # this is an activation function
            nn.Linear(512, 10)  # there are 10 output units (classes)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits  # this is the "raw" output (no softmax yet)


def epoch_train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()  # setting the internal "train" mode e.g. apply dropout and batch normalization
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)  # to gpu if possible (data always comes from cpu)

        # Compute prediction error
        pred = model(X)  # this calls NeuralNetwork.forward(X)
        loss = loss_fn(pred, y)  # loss function applies softmax internally

        # Backpropagation
        # The order is actually important!
        optimizer.zero_grad()  # erase gradients
        loss.backward()  # compute gradients
        optimizer.step()  # update parameters

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def epoch_test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    # turn-off "features" that should be only applied during training
    # e.g. dropout and batch normalization
    model.eval()
    test_loss, correct = 0, 0
    """
    Disabling gradient calculation is useful for inference, when you are sure
    that you will not call :meth:`Tensor.backward()`. It will reduce memory
    consumption for computations that would otherwise have `requires_grad=True`.

    In this mode, the result of every computation will have
    `requires_grad=False`, even when the inputs have `requires_grad=True`.
    """
    # no gradients to be computed during test (faster inference, less memory, no accidental change of model parameters)
    # if you do call backward here, then you train on the test data!
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            # argmax over the logits (same as on softmax)
            correct += (torch.argmax(pred, dim=1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def perform_training(data_dir, ckpts_dir, epochs=3, batch_size=64):
    training_data = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=ToTensor())
    test_data = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=ToTensor())

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    model = NeuralNetwork().to(device)

    loss_fn = nn.CrossEntropyLoss(weight=torch.ones(10))  # all classes have the same weight here
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0,
                                weight_decay=1e-6)  # add the model parameters to optimize

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        epoch_train(train_dataloader, model, loss_fn, optimizer, device)
        epoch_test(test_dataloader, model, loss_fn, device)
    print("Done!")

    # Save checkpoints via PyTorch
    torch.save(model.state_dict(), os.path.join(ckpts_dir, "model.pth"))
    print("Saved PyTorch Model State to model.pth")


def perform_prediction(data_dir, ckpts_dir):
    test_data = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=ToTensor())

    # Load checkpoints via PyTorch
    model = NeuralNetwork()
    model.load_state_dict(torch.load(os.path.join(ckpts_dir, "model.pth")))

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
    x, y = test_data[0][0], test_data[0][1]
    with torch.no_grad():
        pred = model(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')
