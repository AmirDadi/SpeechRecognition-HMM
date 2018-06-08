from data_reader.digits import load_dataset
from hmmlearn import hmm
import numpy as np
from sklearn.externals import joblib
import warnings
import torch
import torch.nn as nn
from AnnHmm import ANNHMM
import torch.nn.functional as F
from torch.autograd import Variable
import itertools
import numpy as np
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class ANN(nn.Module):
    def __init__(self, criterion=nn.CrossEntropyLoss, optimizer=torch.optim.Adam, learning_rate=0.01):
        super(ANN, self).__init__()
        self.layer1 = nn.Linear(13, 8)
        self.optimizer = optimizer(self.parameters(), lr=learning_rate)
        self.criterion = criterion()


    def forward(self, x):
        x = self.layer1(x)
        return F.log_softmax(x)

    def train(self, inputs, labels):
        inputs = Variable(torch.from_numpy(inputs))
        labels = Variable(torch.from_numpy(labels))
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.data.numpy()[0]

model = ANN()
warnings.filterwarnings("ignore", category=DeprecationWarning)
data = load_dataset()

seq_lengths = [[len(seq) for seq in data[i].train] for i in range(10)]
concatenate_data = [np.concatenate(data[i].train, axis=0) for i in range(10)]
#
# model = ANNHMM(model, n_components=8, n_mix=3, algorithm='viterbi', covariance_type="diag", tol=0.0001)
# l = np.sum(seq_lengths[0][:100])
# model.fit(np.array(concatenate_data[0][:l], dtype='float32'), seq_lengths[0][:100])

models = []
for i in range(2):
    model = ANN()
    ann_hmm = ANNHMM(model, n_components=8, n_mix=3, algorithm='viterbi', covariance_type="diag", tol=0.0001)
    ann_hmm.fit(np.array(concatenate_data[i], dtype='float32'), seq_lengths[i])
    models.append(ann_hmm)
test_results = [[0 for i in range(10)] for j in range(10)]
train_results = [[0 for i in range(10)] for j in range(10)]
#
for i in range(2):
    print(i)
    total = 0.0
    correct = 0.0
    for sample in data[i].train[:200]:
        total = total + 1
        scores = np.array([model.score(np.array(sample, dtype='float32')) for model in models])
        best_class = np.argmax(scores)
        test_results[i][best_class] = test_results[i][best_class] + 1
        if best_class == i:
            correct = correct + 1
    print(correct/total)
#
#
print(test_results)
plot_confusion_matrix(np.array(test_results), classes=[i for i in range(10)],
                      title='Confusion matrix, without normalization')
plt.figure()
# print(train_results)


# Train with hmmgmm
# align
