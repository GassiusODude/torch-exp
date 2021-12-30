"""MNIST database of handwritten digits

Based on https://nextjournal.com/gkoehler/pytorch-mnist
"""
import torch
import torchvision
from matplotlib import pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


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
    batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('files/', train=False, download=True,
        transform=im_to_tensor),
    batch_size=batch_size_test, shuffle=True)

log_interval = 10




# ---------------------------  design model  --------------------------------

class MyModel(nn.Module):
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
        return F.log_softmax(x)
    
# ---------------------------  train model  ---------------------------------
def train(model, epoch, device=torch.device('cpu')):
    #model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # move data to device
        data = data.to(device)
        target = target.to(device)

        # apply current network to generate prediction
        output = model(data)

        # compute neg log likelihood loss
        optimizer.zero_grad()
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        # print training loss (based on log interval)
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

        # track training loss
        train_losses.append(loss.item())
        train_counter.append(
            (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
        
        # save network and optimizer state
        torch.save(model.state_dict(), 'results/model.pth')
        torch.save(optimizer.state_dict(), 'results/optimizer.pth')

def test(model, device=torch.device('cpu')):
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
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()

        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--epochs", default=3, type=int, help="Num epochs")
    parser.add_argument("--lr", default=1e-2, type=float, help="Learn rate")
    parser.add_argument("--momentum", default=0.5, type=float, help="Momentum")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    train_losses = []
    train_counter = []
    test_losses = []
    test_counter =\
        [i * len(train_loader.dataset) for i in range(args.epochs +1 )]
        
    # -------------------- initialize model and optimizer  ------------------
    network = MyModel()
    network.to(device)
    optimizer = optim.SGD(
        network.parameters(), lr=args.lr, momentum=args.momentum)

    test(network, device)
    for epoch in range(1, args.epochs + 1):
        train(network, epoch, device)
        test(network, device)

    fig = plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.show()
