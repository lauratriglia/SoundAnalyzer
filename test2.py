import os
import numpy as np
import torch

from pydub import AudioSegment
from pydub.utils import make_chunks
import torchaudio
from speechbrain.pretrained import EncoderClassifier

# make_chunk-> the audio file is divided in chunk of 1 sec, and from each chunk an
# embedding of the chunk is created


def __init__(self):
    self.db_embeddings_audio = None


def predict_speaker(self, embedding):
    score, speaker_name = self.db_embeddings_audio.get_speaker_db_scan(embedding)

    if score == -1:
        speaker_name = "unknown"

    self.db_embeddings_audio.excluded_entities = []
    print("Predicted speaker name is {} with score {}".format(speaker_name, score))

    return speaker_name, float(score)


def get_max_distances(self, emb, thr=None):
    if thr is None:
        thr = self.threshold
    list_distance = []
    label_list = []
    print("Data dictionary size {}".format(len(self.data_dict)))
    for speaker_label, list_emb in self.data_dict.items():
        if speaker_label not in self.excluded_entities:
            for person_emb in list_emb:
                dist = self.similarity_func(torch.from_numpy(person_emb), emb).numpy()
                if dist > thr:
                    list_distance.append(dist[0])
                    label_list.append(speaker_label)

    return list_distance, label_list


def get_speaker_db_scan(self, emb, thr=None):
    distances, labels = self.get_max_distances(emb, thr)
    if len(distances) == 0:
        return -1, -1

    n = len(distances) if len(distances) < self.n_neighbors else self.n_neighbors
    try:
        max_dist_idx = np.argpartition(distances, -n)[-n:]
        count = dict()
        for i in max_dist_idx:
            if labels[i] not in count.keys():
                count[labels[i]] = [0, distances[i]]
                continue
            count[labels[i]][0] += 1
            count[labels[i]][1] += distances[i]

        max_count = 0
        max_dist = 0
        final_label = ''
        for key, value in count.items():
            if value[0] >= max_count and value[1] >= max_dist:
                max_count = value[0]
                max_dist = value[1]
                final_label = key

        self.excluded_entities.append(final_label)
        return max_dist / max_count, final_label

    except Exception as e:
        return -1, -1


def make_chunk(file_name):
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
    audio_file = AudioSegment.from_file("1641898112.818368_1641898266.2594984.wav", "wav")
    chunk_length = 1000  # pydub calculates in millisec
    chunks = make_chunks(audio_file, chunk_length)  # make chunks of 1 sec

    for i, chunk in enumerate(chunks):
        chunk_name = './chunks/' + "chunk{0}.wav".format(i)
        signal, fs = torchaudio.load(chunk_name)
        embeddings = classifier.encode_batch(signal)
        embeddings = embeddings.squeeze(axis=0)
        output_probs, score, index, text_lab = classifier.classify_batch(signal)
        print("Predicted: " + text_lab[0])
        print(embeddings)
        predict_speaker(chunk_name, embeddings)



all_file_names = os.listdir()
try:
    os.makedirs('chunks')
except:
    pass
for each_file in all_file_names:
    if('.wav' in each_file):
        make_chunk(each_file)