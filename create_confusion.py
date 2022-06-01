import os
import numpy as np
import random
import math
import torch
import csv
from pydub import AudioSegment
from pydub.utils import make_chunks
import wave
import librosa
import torchaudio
from speechbrain.pretrained import EncoderClassifier
from speaker_embeddings import EmbeddingsHandler
from emb_creation1 import predict_speaker
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def main():
    main_dir = 'DB_s'
    all_file_names = os.listdir(main_dir)
    list_data = []
    list_all_name = []

    for subdir in all_file_names:
        if os.path.isdir(main_dir + '/' + subdir):
            dir_names = os.listdir(main_dir + '/' + subdir)
            print(subdir)
            list_all_name.append(subdir)
        for subdir1 in dir_names:
            if os.path.isdir(main_dir + '/' + subdir + '/' + subdir1):
                embs = np.load(main_dir + '/' + subdir + '/' + subdir1 + '/' + 'chunk.npy').squeeze()
                # print(f"EMB outside: {embs}")
                # print(embs.shape)
                if len(embs.shape) > 1:
                    for emb in embs:
                        list_data.append([subdir, emb])
                else:
                    list_data.append([subdir, embs])
    #a = np.ndarray(shape=(0,0))
    #list_data.append(["Silence_audio", a])
    # print(list_data)
    random.shuffle(list_data)
    perc_train = 0.7
    n_train = math.ceil(0.7 * len(list_data))
    n_test = len(list_data) - n_train
    train_set = list_data[0:n_train]
    test_set = list_data[n_train:]
    list_pred_name = []
    list_true_name = []

    ls = EmbeddingsHandler(os.path.join(os.getcwd(), 'DB_s'), n_neighbors=4, train_set=train_set)
    print(list_all_name)
    print(len(list_all_name))
    # print(ls.data_dict)

    for data in test_set:
        true_name = data[0]
        emb = data[1]
        # print(emb)
        # print(test_set)
        # print(embs)
        pred_name = predict_speaker(emb, ls)
        list_pred_name.append(pred_name)
        list_true_name.append(true_name)
    print(list_pred_name)
    print(list_true_name)

    cm = confusion_matrix(list_true_name, list_pred_name)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list_all_name)
    disp.plot()
    plt.show()


if __name__ == '__main__':
    main()