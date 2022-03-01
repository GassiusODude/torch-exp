"""MNIST database of handwritten digits

Use 2 layers of fully connected dense network to perform multiclass
classification
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


# ---------------------------  design model  --------------------------------
class DnnModel1(nn.Module):
    """
    """
    def __init__(self, num_classes=10, mid=200):
        super(DnnModel1, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784, mid),
            nn.LeakyReLU(0.02),
            nn.Linear(mid, num_classes),
            nn.LeakyReLU(0.02),
        )

    def forward(self, x):
        x = x.view(-1, 784)
        return self.model.forward(x)

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
        loss_func =nn.CrossEntropyLoss()
        loss = loss_func(output, target)
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

            # predict and track correct
            pred = torch.argmax(output, 1)
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
    comment = "%s_opt%d_lr%0.4f" % ("DNN1", args.opt, args.lr)
    writer = SummaryWriter(comment=comment)

    # -------------------- initialize model and optimizer  ------------------
    network = DnnModel1()
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