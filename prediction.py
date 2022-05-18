import os
import numpy as np
import torch
import csv
from pydub import AudioSegment
from pydub.utils import make_chunks
import torchaudio
from speechbrain.pretrained import EncoderClassifier
from speaker_embeddings import EmbeddingsHandler
from embeddings_creation import EmbeddingsCreation


def predict_speaker(embeddings):

    ls = EmbeddingsHandler(os.path.join(os.getcwd(), 'DB_Final'), n_neighbors=4)
    score, speaker_name = ls.get_speaker_db_scan(embeddings)

    if score == -1:
        speaker_name = "unknown"

    ls.excluded_entities = []
    print("Predicted speaker name is {} with score {}".format(speaker_name, score))

    return speaker_name, float(score)


main_dir = 'DB_Final'
all_file_names = os.listdir(main_dir)
print(all_file_names)

for subdir in all_file_names:
    if os.path.isdir(main_dir + '/' + subdir):
        dir_names = os.listdir(main_dir + '/' + subdir)
    for subdir1 in dir_names:
        if os.path.isdir(main_dir+'/'+subdir+'/'+subdir1):
            files = os.listdir(main_dir+'/'+subdir+'/'+subdir1)
            for file in files:
                if '.npy' in file:
                    # speaker_emb = lib.make_chunk(main_dir+'/'+subdir+'/'+subdir1+'/'+file)
                    lib = EmbeddingsCreation()
                    speaker_name, score = predict_speaker(lib.embeddings)

