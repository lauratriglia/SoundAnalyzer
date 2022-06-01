import os
import numpy as np
import random
import math
from speaker_embeddings import EmbeddingsHandler
from prediction import predict_speaker
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
                if len(embs.shape) > 1:
                    for emb in embs:
                        list_data.append([subdir, emb])
                else:
                    list_data.append([subdir, embs])

    random.shuffle(list_data)
    perc_train = 0.7
    n_train = math.ceil(perc_train * len(list_data))
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
