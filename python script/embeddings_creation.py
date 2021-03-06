import os
import numpy as np
import torch
import csv
from pydub import AudioSegment
from pydub.utils import make_chunks
import torchaudio
from speechbrain.pretrained import EncoderClassifier
from speaker_embeddings import EmbeddingsHandler


class EmbeddingsCreation:
    def __init__(self):
        self.embeddings = torch.Tensor()

    def make_chunk(self, audio_file):
        file_name = os.path.basename(audio_file)
        dir_name = os.path.dirname(audio_file)
        #print(file_name)
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
            list_emb.append(embeddings.numpy())
            # output_probs, score, index, text_lab = classifier.classify_batch(signal)
            # print(embeddings)

        with open(dir_name+'/chunk.npy', 'wb') as f:
            np.save(f, np.array(list_emb))

        self.embeddings=embeddings

        #predict_speaker(embeddings)
        return embeddings

    main_dir = 'DB_Final'
    all_file_names = os.listdir(main_dir)
    print(all_file_names)

    for subdir in all_file_names:
        if os.path.isdir(main_dir + '/' + subdir):
            dir_names = os.listdir(main_dir + '/' + subdir)
        for subdir1 in dir_names:
            if os.path.isdir(main_dir+'/'+subdir+'/'+subdir1):
                files = os.listdir(main_dir+'/'+subdir+'/'+subdir1)
                # predict_speaker(np.load(main_dir+'/'+subdir+'/'+subdir1+'/'+'chunk.npy').squeeze()[0])
               # for file in files:
               #     if '.wav' in file:
                    # make_chunk(main_dir+'/'+subdir+'/'+subdir1+'/'+file)
