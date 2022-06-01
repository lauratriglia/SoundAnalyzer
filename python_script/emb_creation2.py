import os
import numpy as np
import torch
import csv
from pydub import AudioSegment
from pydub.utils import make_chunks
import pyaudio
import wave
import torchaudio
from speechbrain.pretrained import EncoderClassifier
from speaker_embeddings import EmbeddingsHandler


def emb_creation(audio_file):
    #file_name = os.path.basename(audio_file)
    dir_name = os.path.dirname(audio_file)
    print(audio_file)
    sound = wave.open(audio_file)
    # print(file_name)
    #print(dir_name)
    list_chunk = []
    nb_samples_received = 0
    list_emb = []
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
    # chunk_length = 1000  # pydub calculates in millisec
    # chunks = make_chunks(audio_file, chunk_length)  # make chunks of 1 sec

    chunk = np.zeros((sound.getnchannels(), sound.getsampwidth()), dtype=np.float32)
    nb_samples_received += sound.getsampwidth()
    for c in range(sound.getnchannels()):
        for i in range(sound.getsampwidth()):
            chunk = sound.getparams()

    list_chunk.append(chunk)
    for i, chunk in list_chunk:
        chunk_name = './chunks/' + "chunk{0}.wav".format(i)
        signal, fs = torchaudio.load(chunk_name)
        embeddings = classifier.encode_batch(signal)
        embeddings = embeddings.squeeze(axis=0)
        list_emb.append(embeddings.numpy())
        # output_probs, score, index, text_lab = classifier.classify_batch(signal)
        # print(embeddings)

    with open(dir_name+'/chunk.npy', 'wb') as f:
        np.save(f, np.array(list_emb))

    # predict_speaker(embeddings)
    return embeddings


def predict_speaker(embeddings):

    ls = EmbeddingsHandler(os.path.join(os.getcwd(), 'DB_Final'), n_neighbors=4)
    score, speaker_name = ls.get_speaker_db_scan(embeddings)

    if score == -1:
        speaker_name = "unknown"

    ls.excluded_entities = []
    print("Predicted speaker name is {} with score {}".format(speaker_name, score))

    return speaker_name, float(score)


main_dir = 'DB_Final (copy)'
all_file_names = os.listdir(main_dir)
# print(all_file_names)


for subdir in all_file_names:
    if os.path.isdir(main_dir + '/' + subdir):
        dir_names = os.listdir(main_dir + '/' + subdir)
    for subdir1 in dir_names:
        if os.path.isdir(main_dir+'/'+subdir+'/'+subdir1):
            files = os.listdir(main_dir+'/'+subdir+'/'+subdir1)
            # predict_speaker(np.load(main_dir+'/'+subdir+'/'+subdir1+'/'+'chunk.npy').squeeze()[0])
            for file in files:
                if '.wav' in file:
                  emb_creation(main_dir+'/'+subdir+'/'+subdir1+'/'+file)
