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
    dataset = CIFAR10(root='..', download=True, train=training,
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

    def forward(self, x, ts):
        # The input image is in standard "BCHW" format
        # To use an MLP, we reshape it to a 1D vector
        batch_size, channels, height, width = x.shape
        x = x.reshape((batch_size, -1))

        # A two layer MLP
        x = self.fc1(x)
        x = F.relu(x)
        ts.collect('Layer 1 Activation Mean', x.mean())
        ts.collect('Layer 1 Activation Variance', x.var(0).mean())
        ts.collect('Dead Neurons', torch.sum(x == 0))
        x = self.fc2(x)
        ts.collect('Layer 2 Activation Mean', x.mean())
        ts.collect('Layer 2 Activation Variance', x.var(0).mean())
        x = F.softmax(x, dim=1)
        return x


netC = Classifier().to(device)

optimizerC = optim.Adam(netC.parameters(), lr=opt.lr)

total_batches = len(train_dataloader) + len(test_dataloader)
ts = TimeSeries('CIFAR10 Training', opt.epochs * total_batches)

for epoch in range(opt.epochs):
    for data_batch, labels in train_dataloader:
        data_batch = data_batch.to(device)
        labels = labels.to(device)

        netC.zero_grad()
        predictions = netC(data_batch, ts)
        loss = F.cross_entropy(predictions, labels)
        loss.backward()
        optimizerC.step()

        pred_confidence, pred_argmax = predictions.max(dim=1)
        correct = torch.sum(pred_argmax == labels)
        accuracy = float(correct) / len(data_batch)

        ts.collect('Training Loss', loss)
        ts.collect('Training Accuracy', accuracy)
        ts.print_every(n_sec=4)

    for data_batch, labels in test_dataloader:
        data_batch = data_batch.to(device)
        labels = labels.to(device)

        predictions = netC(data_batch, ts)
        pred_confidence, pred_argmax = predictions.max(dim=1)
        correct = torch.sum(pred_argmax == labels)

        ts.collect('Testing Loss', loss)
        ts.collect('Testing Accuracy', float(correct) / len(data_batch))
        ts.print_every(n_sec=4)
print(ts)
print('Final results: {} correct out of {}'.format(correct, len(test_dataloader.dataset)))
