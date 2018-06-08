from collections import defaultdict
import numpy as np


class Dataset:
    def __init__(self, dim):
        self.train = []
        self.test = []
        self.dim = dim


def read_block(data_file):
    observations = []
    while True:
        line = data_file.readline()
        if len(line.strip()) == 0:
            if len(observations) > 0:
                break
            else:
                continue
        frame = [float(a) for a in line.split(' ')]
        observations.append(frame)
    observations = np.asarray(observations)

    return observations


def load_dataset():
    res = defaultdict(lambda: Dataset(13))
    with open('./data/Train_Arabic_Digit.txt') as data_file:
        for digit in range(10):
            for block in range(660):
                data = read_block(data_file)
                res[digit].train.append(data)

    with open('./data/Test_Arabic_Digit.txt') as data_file:
        for digit in range(10):
            for block in range(220):
                data = read_block(data_file)
                res[digit].test.append(data)

    return res

