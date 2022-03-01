"""MNIST database of handwritten digits


References
----------
.. [1] G. Koehler, "MNIST Handwritten Digit Recognition in PyTorch",
    2020, Available:
    https://nextjournal.com/gkoehler/pytorch-mnist
.. [2] PyTorch, "Torch.Utils.Tensorboard",
    https://pytorch.org/docs/stable/tensorboard.html
"""
from tqdm import tqdm
import torch
import torchvision
from matplotlib import pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

batch_size_train = 64
batch_size_test = 1000

# specify transform to use on the loaded images
im_to_tensor = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,))
])

# -----------  create data loader for training and testing sets  ------------
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
       'files/', train=True, download=True,
        transform=im_to_tensor),
    batch_size=batch_size_train, shuffle=True, num_workers=6)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('files/', train=False, download=True,
        transform=im_to_tensor),
    batch_size=batch_size_test, shuffle=True, num_workers=6)

log_interval = 10




# ---------------------------  design model  --------------------------------
class MyModel(nn.Module):
    """
    Model is from [1_].

    My interpretation of what it is doing is applying 5x5 kernel and
    downsampling by 2.  Followed by a second stage of kernel and
    downsample.  The output is then pushed through a dense layer
    from 320 samples -> 50, followed by a second dense layer bringing
    the samples down from 50 to 10.  The last layer is a softmax
    to choose the highest of 10 classification labels (0-9)
    """
    def __init__(self):
        super(MyModel, self).__init__()
        # expand 10 fold and apply kernel of size 5
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)

        # expand by 20 times
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)

        # prevent overfitting on training data
        self.conv2_drop = nn.Dropout2d()

        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        # input image is 28 x 28, single channel
        #   thus x is [batch, 1, 28, 28]
        x = F.relu(F.max_pool2d(self.conv1(x), 2))

        # new size is [batch, 10, 12, 12]
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

        # [batch, 20, 4, 4]
        x = x.view(-1, 320)

        # [batch, 320]
        x = F.relu(self.fc1(x))

        # [batch x 50]
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        # [batch, 10] (10 classes from 0 - 9)
        return F.log_softmax(x, dim=1)

# ---------------------------  train model  ---------------------------------
def train(model, epoch, device=torch.device('cpu'), writer=None):
    #model.train()

    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        # move data to device
        data = data.to(device)
        target = target.to(device)

        # apply current network to generate prediction
        output = model(data)

        # compute neg log likelihood loss

        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

    if writer:
        writer.add_scalar("Loss/train", loss.item(), epoch)

    # save network and optimizer state
    torch.save(model.state_dict(), 'results/model.pth')
    torch.save(optimizer.state_dict(), 'results/optimizer.pth')

def test(model, epoch, device=torch.device('cpu'), writer=None):
    #model.test()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            # move data to device
            data = data.to(device)
            target = target.to(device)

            # apply model on input data
            output = model(data)

            # calculate loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()

        acc = 100. * correct / len(test_loader.dataset)

        if writer:
            writer.add_scalar("Accuracy/test", acc, epoch)


if __name__ == "__main__":
    # -------------------------  parse arguments  ---------------------------
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--epochs", default=3, type=int, help="Num epochs")
    parser.add_argument("--lr", default=1e-2, type=float, help="Learn rate")
    parser.add_argument("--momentum", default=0.5, type=float, help="Momentum")
    parser.add_argument(
        "--opt", default=0, choices=range(2), type=int, help="0=SGD, 1=Adam")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    # setup device
    if args.device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    # setup summary writer
    comment = "%s_%d_%0.4f" % ("Model1", args.opt, args.lr)
    writer = SummaryWriter(comment=comment)

    # -------------------- initialize model and optimizer  ------------------
    network = MyModel()
    network.to(device)
    if args.opt == 0:
        optimizer = optim.SGD(
            network.parameters(), lr=args.lr, momentum=args.momentum)

    else:
        optimizer = optim.Adam(network.parameters(), lr=args.lr)

    test(network, 0, device, writer=writer)
    for epoch in range(1, args.epochs + 1):
        train(network, epoch, device,writer=writer)
        test(network, epoch, device, writer=writer)

    writer.close()