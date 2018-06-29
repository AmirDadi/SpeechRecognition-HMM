from data_reader.sounds import read_data
import torch.nn as nn
from torch.autograd import Variable
import torch
import numpy as np
import torch.nn.functional as F
from warpctc_pytorch import CTCLoss

dataset = read_data()


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(41, 11), stride=(2, 2),
                      padding=(0, 10)),
            nn.BatchNorm2d(8),
            nn.Hardtanh()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=(21, 11), stride=(2, 2)),
            nn.BatchNorm2d(4),
        )
        self.layer3 = nn.RNN(input_size=7098*4, hidden_size=8, num_layers=2, nonlinearity='relu', batch_first=True)
        self.fcc = nn.Linear(8, 4)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = F.hardtanh(out)
        out = np.reshape(out, (1, -1, 4))
        print(np.shape(out))
        out = self.layer3(out)

        return self.softmax(self.fcc(out))



def train():
    cnn = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)
    data = dataset.train_data
    for sample in data:
        spectogram = Variable(torch.from_numpy(sample['spectogram']).unsqueeze(0).unsqueeze(0))
        labels = Variable(torch.from_numpy(sample['sentence']))
        optimizer.zero_grad()
        outputs = cnn(spectogram)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


train()