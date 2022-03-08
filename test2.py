import os
import numpy as np

from pydub import AudioSegment
from pydub.utils import make_chunks
import torchaudio
from speechbrain.pretrained import EncoderClassifier


def make_chunk(file_name):
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
    audio_file = AudioSegment.from_file("no_silence_audio_test.wav", "wav")
    chunk_length = 1000  # pydub calculates in millisec
    chunks = make_chunks(audio_file, chunk_length)  # make chunks of 1 sec

    for i, chunk in enumerate(chunks):
        chunk_name = './chunks/' + "chunk{0}.wav".format(i)
        print("exporting", chunk_name)
        chunk.export(chunk_name, format="wav")
        signal, fs = torchaudio.load(chunk_name)
        embeddings = classifier.encode_batch(signal)
        embeddings = embeddings.squeeze(axis=0)
        output_probs, score, index, text_lab = classifier.classify_batch(signal)
        print("Predicted: " + text_lab[0])
        print(embeddings)


all_file_names = os.listdir()
try:
    os.makedirs('chunks')
except:
    pass
for each_file in all_file_names:
    if('.wav' in each_file):
        make_chunk(each_file)