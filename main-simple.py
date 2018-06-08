from data_reader.digits import load_dataset
from hmmlearn import hmm
import numpy as np
from sklearn.externals import joblib
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

data = load_dataset()
seq_lengths = [[len(seq) for seq in data[i].train] for i in range(10)]
concatenate_data = [np.concatenate(data[i].train, axis=0) for i in range(10)]

models = [hmm.GMMHMM(n_components=8, n_mix=3, algorithm='viterbi', covariance_type="diag").fit(concatenate_dataset, seq_length)
          for (concatenate_dataset, seq_length) in zip(concatenate_data, seq_lengths)]
joblib.dump(models, 'models.pkl')

test_results = [[i for i in range(10)] for j in range(10)]
train_results = [[i for i in range(10)] for j in range(10)]

for i in range(10):
    print(i)
    total = 0.0
    correct = 0.0
    for sample in data[i].test:
        total = total + 1
        scores = np.array([model.score(sample) for model in models])
        best_class = np.argmax(scores)
        test_results[i][best_class] = test_results[i][best_class] + 1
        if best_class == i:
            correct = correct + 1
    print(correct/total)


print(test_results)
# print(train_results)


# Train with hmmgmm
# align
