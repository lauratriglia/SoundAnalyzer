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


def make_chunk(audio_file_name):
    file_name = os.path.basename(audio_file_name)
    dir_name = os.path.dirname(audio_file_name)
    list_name = os.listdir(dir_name)
    # print(list_name)
    # print(audio_file_name)
    # print(file_name)
    print(dir_name)
    list_emb = []
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
    chunk_length = 1000  # pydub calculates in millisec
    # sound = wave.open(audio_file)
    # print(librosa.get_duration(filename=audio_file_name))

    if librosa.get_duration(filename=audio_file_name) < chunk_length/1000:
       return

    audio_file = AudioSegment.from_file(audio_file_name, "wav")
    chunks = make_chunks(audio_file, chunk_length)  # make chunks of 1 sec
    chunks_dir = os.path.join(dir_name, 'n_chunks')
    # print(chunks_dir)

    for i, chunk in enumerate(chunks):
        chunk_name = chunks_dir + '/'+ "chunk{0}.wav".format(i)
        # print(chunk_name)
        chunk.export(chunk_name, format="wav")
        # print(librosa.get_duration(filename= chunk_name))
        if librosa.get_duration(filename= chunk_name) >= chunk_length/1000:
            signal, fs = torchaudio.load(chunk_name)
            embeddings = classifier.encode_batch(signal)
            embeddings = embeddings.squeeze()
            list_emb.append(embeddings.numpy())

    print("###")
    print(len(chunks))
    print(len(list_emb))
    print("###")
    with open(dir_name+'/chunk.npy', 'wb') as f:
        np.save(f, np.array(list_emb))


def predict_speaker(embeddings, ls):

    # ls = EmbeddingsHandler(os.path.join(os.getcwd(), 'DB_s'), n_neighbors=4)
    speaker_name = ls.get_speaker_db_scan(embeddings)

    # if score == -1:
    #     speaker_name = "unknown"

    ls.excluded_entities = []
    print("Predicted speaker name is {}".format(speaker_name))

    return speaker_name

def main():

    main_dir = 'DB_Final'
    all_file_names = os.listdir(main_dir)
    for subdir in all_file_names:
        if os.path.isdir(main_dir + '/' + subdir):
            dir_names = os.listdir(main_dir + '/' + subdir)
            # print(subdir)
        for subdir1 in dir_names:
            if os.path.isdir(main_dir+'/'+subdir+'/'+subdir1):
                files = os.listdir(main_dir+'/'+subdir+'/'+subdir1)
                embs = np.load(main_dir+'/'+subdir+'/'+subdir1+'/'+'chunk.npy').squeeze()
                # print(f"EMB outside: {embs}")
                print(subdir1)
                # predict_speaker(np.load(main_dir+'/'+subdir+'/'+subdir1+'/'+'chunk.npy'))
                # print(embs)
                # for emb in embs:
                #    predict_speaker(emb)
                for file in files:
                     if '.wav' in file:
                      make_chunk(main_dir+'/'+subdir+'/'+subdir1+'/'+file)

if __name__ == '__main__':
    main()