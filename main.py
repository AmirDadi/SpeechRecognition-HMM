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

number_of_tests = 4

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


def iter_from_X_lengths(X, lengths):
    if lengths is None:
        yield 0, len(X)
    else:
        n_samples = X.shape[0]
        end = np.cumsum(lengths).astype(np.int32)
        start = end - lengths
        if end[-1] > n_samples:
            raise ValueError("more than {0:d} samples in lengths array {1!s}"
                             .format(n_samples, lengths))

        for i in range(len(lengths)):
            yield start[i], end[i]

def train_dnn(model, features, labels, other_features):
    # for feature, label in zip(features, labels):
    model.train(features, labels)
    # for feature in other_features:
    model.train(other_features, np.repeat(5, len(other_features)))
    return model


def get_random_sample_of_others(concatenate_data, j):
    random_sample = []
    count = 0
    for i, data in enumerate(concatenate_data):
        if i != j:
            for feature in data:
                if np.random.rand() < 0.01:
                    random_sample.append(feature)
                    count += 1
    print(count)
    return np.array(random_sample)


class ANN(nn.Module):
    def __init__(self, criterion=nn.CrossEntropyLoss, optimizer=torch.optim.RMSprop, learning_rate=0.001):
        super(ANN, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(13, 20)
        # self.fc2 = nn.Linear(50, 20)
        self.fc3 = nn.Linear(20, 6)
        self.softmax = nn.Softmax(dim=1)

        self.optimizer = optimizer(self.parameters(), lr=learning_rate)
        self.criterion = criterion()

    def forward(self, x):
        x = x.float()
        x = self.fc1(x)
        x = self.relu(x)
        # x = self.fc2(x)
        # x = self.relu(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x

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

seq_lengths = [[len(seq) for seq in data[i].train] for i in range(number_of_tests)]
concatenate_data = [np.concatenate(data[i].train, axis=0) for i in range(number_of_tests)]
#
# model = ANNHMM(model, n_components=8, n_mix=3, algorithm='viterbi', covariance_type="diag", tol=0.0001)
# l = np.sum(seq_lengths[0][:100])
# model.fit(np.array(concatenate_data[0][:l], dtype='float32'), seq_lengths[0][:100])

models = []
lab = [0, 0, 0, 0, 0]

for i in range(number_of_tests):
    model = ANN()
    ann_hmm = ANNHMM(model, n_components=5, n_mix=3, algorithm='viterbi', covariance_type="diag", tol=0.0001)
    ann_hmm.fit(np.array(concatenate_data[i], dtype='float32'), seq_lengths[i])
    models.append(ann_hmm)

# for i, model in enumerate(models):
#     feature = concatenate_data[i]
#     seq_len_of_features = seq_lengths[i]
#     labels = model.predict(feature, seq_len_of_features)
#     other_features = get_random_sample_of_others(concatenate_data, i)
#     train_dnn(model.Ann_model, feature, labels, other_features)
#     model.is_gmm = False
#     print(i)
#     model.fit(np.array(concatenate_data[i], dtype='float32'), seq_lengths[i])

test_results = [[0 for i in range(10)] for j in range(10)]
train_results = [[0 for i in range(10)] for j in range(10)]
#


sampled_scores = [0, 0, 0, 0]
for i in [0, 1,2,3]:
    print(i)
    total = 0.0
    correct = 0.0
    for sample in data[i].train[:200]:
        total = total + 1
        scores = np.array([model.score(np.array(sample, dtype='float32')) for model in models])
        sampled_scores += scores

print(sampled_scores)

for i in [0,1,2,3]:
    print(i)
    total = 0.0
    correct = 0.0
    for sample in data[i].train[:200]:
        total = total + 1
        scores = np.array([model.score(np.array(sample, dtype='float32')) for model in models])
        scores /= sampled_scores
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
