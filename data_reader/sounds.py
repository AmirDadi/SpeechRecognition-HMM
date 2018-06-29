import glob, os
import soundfile as sf
from scipy import signal
import numpy as np

class DataSet:
    def __init__(self):
        self.train_data = []
        self.train_labels = []
        self.test = []

os.chdir("./data/LibriSpeech/dev-clean-2/")
count = 0
dirs = []


def map_int(sentence):
    res = []
    for i in sentence:
        if i == ' ':
            res.append(1)
        else:
            res.append(ord('z') - ord(i))
    return np.array(res)

def read_data():
    ds = DataSet()
    for file in glob.glob("**/**/*.txt"):
        with open(file) as data_file:
            line = data_file.readline()
            while line:
                name, sentence = line.split(' ', 1)
                ds.train_labels.append({
                    'name': name,
                    'sentence': sentence
                })
                line = data_file.readline()

    for file in glob.glob("**/**/*.flac"):
        name = file.split('/')[-1].split('.')[0]
        data, _ = sf.read(file)
        sentence = ''
        window_size = 20
        step_size = 10
        sample_rate = 16000
        nperseg = int(round(window_size * sample_rate / 1e3))
        noverlap = int(round(step_size * sample_rate / 1e3))
        _, _, spec = signal.spectrogram(data,
                                        fs=sample_rate,
                                        window='hann',
                                        nperseg=nperseg,
                                        noverlap=noverlap,
                                        detrend=False)
        for label in ds.train_labels:
            if name == label['name']:
                sentence = label['sentence']
        ds.train_data.append({
            'name': name,
            'wave': data.astype(np.float32),
            'sentence': map_int(sentence),
            'spectogram': spec.astype(np.float32)
        })
    return ds


