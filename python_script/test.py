import os
import numpy as np
import torch
import csv
from pydub import AudioSegment
from pydub.utils import make_chunks
import torchaudio
from speechbrain.pretrained import EncoderClassifier


def make_chunk(audio_file):
    file_name = os.path.basename(audio_file)
    dir_name = os.path.dirname(audio_file)
    print(file_name)
    print(dir_name)
    list_emb = []
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
    chunk_length = 1000  # pydub calculates in millisec
    chunks = make_chunks(audio_file, chunk_length)  # make chunks of 1 sec

    for i, chunk in enumerate(chunks):
        chunk_name = './chunks/' + "chunk{0}.wav".format(i)
        signal, fs = torchaudio.load(chunk_name)
        embeddings = classifier.encode_batch(signal)
        embeddings = embeddings.squeeze(axis=0)
        list_emb.append(embeddings.tolist())
        output_probs, score, index, text_lab = classifier.classify_batch(signal)
        print(output_probs, score, index, text_lab)
        print("Predicted: " + text_lab[0])
        # print(embeddings)

    with open(dir_name+'/chunk.csv', 'w', encoding='UTF8') as f:
        # create the csv writer
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        for emb in list_emb:
            # write a row to the csv file
            writer.writerow(emb)

main_dir = '20220110-143437'
all_file_names = os.listdir(main_dir)
print(all_file_names)

for subdir in all_file_names:
    if os.path.isdir(main_dir+'/'+subdir):
        files = os.listdir(main_dir+'/'+subdir)
        for file in files:
            if('.wav' in file):
                make_chunk(main_dir+'/'+subdir+'/'+file)

