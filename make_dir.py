import os
import numpy as np
import torch
import csv
from pydub import AudioSegment
from pydub.utils import make_chunks
import wave
import librosa
import torchaudio
from speechbrain.pretrained import EncoderClassifier
from speaker_embeddings import EmbeddingsHandler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

main_dir = 'DB_Final'
all_file_names = os.listdir(main_dir)


for subdir in all_file_names:
    if os.path.isdir(main_dir + '/' + subdir):
        dir_names = os.listdir(main_dir + '/' + subdir)
        print(subdir)
    for subdir1 in dir_names:
        if os.path.isdir(main_dir+'/'+subdir+'/'+subdir1):
            files = os.listdir(main_dir+'/'+subdir+'/'+subdir1)
            print(subdir1)
            os.mkdir(main_dir+'/'+subdir+'/'+subdir1 + '/' + 'n_chunks')