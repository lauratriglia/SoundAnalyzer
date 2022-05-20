import os
import numpy as np
import torch
import csv
from pydub import AudioSegment
from pydub.utils import make_chunks
import torchaudio
from speechbrain.pretrained import EncoderClassifier
from speaker_embeddings import EmbeddingsHandler


def make_chunk(audio_file):
    file_name = os.path.basename(audio_file)
    dir_name = os.path.dirname(audio_file)
    print(audio_file)
    # print(file_name)
    # print(dir_name)
    list_emb = []
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
    chunk_length = 1000  # pydub calculates in millisec
    chunks = make_chunks(audio_file, chunk_length)  # make chunks of 1 sec

    for i, chunk in enumerate(chunks):
        chunk_name = './chunks/' + os.path.splitext(file_name)[0] + "_chunk{0}.wav".format(i)
       # chunk.export(chunk_name, format="wav")
        #print(chunk_name)
        signal, fs = torchaudio.load(chunk)
        embeddings = classifier.encode_batch(signal)
        embeddings = embeddings.squeeze(axis=0)
        list_emb.append(embeddings.numpy())
       # print(list_emb)
        # output_probs, score, index, text_lab = classifier.classify_batch(signal)
        # print(embeddings)

    with open(dir_name+'/chunk.npy', 'wb') as f:
        np.save(f, np.array(list_emb))

    #predict_speaker(embeddings)
    return embeddings


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


for subdir in all_file_names:
    if os.path.isdir(main_dir + '/' + subdir):
        dir_names = os.listdir(main_dir + '/' + subdir)
        print(subdir)
    for subdir1 in dir_names:
        if os.path.isdir(main_dir+'/'+subdir+'/'+subdir1):
            files = os.listdir(main_dir+'/'+subdir+'/'+subdir1)
            embs = np.load(main_dir+'/'+subdir+'/'+subdir1+'/'+'chunk.npy').squeeze()
            # print(f"EMB outside: {embs}")
            print(subdir1)
            for emb in embs:
                predict_speaker(emb)
            #for file in files:
            #     if '.wav' in file:
            #      make_chunk(main_dir+'/'+subdir+'/'+subdir1+'/'+file)

#emb = np.load(main_dir+'/'+'Giacomo'+'/'+'2022_01_10'+'/'+'chunk.npy').squeeze()
#print(emb)
#predict_speaker(emb)