from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torchvision.datasets import CIFAR10
from torchvision import transforms

from torch.nn import functional as F
from gnomehat.series import TimeSeries


parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--epochs', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--cuda', default=True, action="store_true", help='Use CUDA (requires GPU)')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')

opt = parser.parse_args()
print(opt)

def get_cifar(training=False):
    # CIFAR stands for Canadian Institute for Advanced Research
    # The CIFAR dataset consists of 32x32 RGB images of 10 categories
    dataset = CIFAR10(root='.', download=True, train=training,
                           transform=transforms.Compose([
                               transforms.Resize(32),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
    return torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True)


train_dataloader = get_cifar(training=True)
test_dataloader = get_cifar(training=False)

device = torch.device("cuda" if opt.cuda else "cpu")


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(32*32*3, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # The input image is in standard "BCHW" format
        # To use an MLP, we reshape it to a 1D vector
        batch_size, channels, height, width = x.shape
        x = x.reshape((batch_size, -1))

        # A two layer MLP
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.softmax(x)
        return x


netC = Classifier().to(device)

optimizerC = optim.Adam(netC.parameters(), lr=opt.lr)


for epoch in range(opt.epochs):
    ts = TimeSeries('CIFAR10 Training', len(train_dataloader))
    for data, labels in train_dataloader:
        data = data.to(device)
        labels = labels.to(device)

        netC.zero_grad()
        predictions = netC(data)
        loss = F.cross_entropy(predictions, labels)
        loss.backward()
        optimizerC.step()

        pred_confidence, pred_argmax = predictions.max(dim=1)
        accuracy = torch.sum(pred_argmax == labels)
        ts.collect('Training Loss', loss)
        ts.collect('Training Accuracy', accuracy)
        ts.print_every(n_sec=1)

    ts_test = TimeSeries('CIFAR10 Testing')
    for data, labels in test_dataloader:
        data = data.to(device)
        labels = labels.to(device)

        predictions = netC(data)
        pred_confidence, pred_argmax = predictions.max(dim=1)
        accuracy = torch.sum(pred_argmax == labels)

        ts_test.collect('Testing Loss', loss)
        ts_test.collect('Testing Accuracy', accuracy)
        ts_test.print_every(n_sec=1)
    print(ts_test)
