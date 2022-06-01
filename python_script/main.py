import sys
import os
import time

from torchvision import transforms
import torch, torchaudio
import yarp
import numpy as np
from speechbrain.pretrained import EncoderClassifier
from voiceRecognition.speaker_embeddings import EmbeddingsHandler

from pydub import AudioSegment
from pydub.utils import make_chunks

import scipy.io.wavfile as wavfile
import scipy
import dlib
import cv2 as cv


def info(msg):
    print("[INFO] {}".format(msg))


class PersonsRecognition(yarp.RFModule):
    """
    Description:
        Class to recognize a person from the audio or the face
    Args:
        input_port  : Audio from remoteInterface, raw image from iCub cameras
    """

    def __init__(self):

        # handle port for the RFModule
        self.module_name = None
        self.handle_port = None
        self.process = False

        # Define vars to receive audio
        self.audio_in_port = None
        self.eventPort = None
        self.is_voice = False

        # Predictions parameters
        self.label_outputPort = None
        self.predictions = []
        self.database = None

        # Speaker module parameters
        self.model_audio = None
        self.dataset_path = None
        self.db_embeddings_audio = None
        self.threshold_audio = None
        self.length_input = None
        self.resample_trans = None
        self.speaker_emb = []

        # Parameters for the audio
        self.sound = None
        self.audio = []
        self.np_audio = None
        self.nb_samples_received = 0
        self.sampling_rate = None

        self.device = None

        self.name = ""
        self.predict = False

    def load_model_audio(self):
        self.resample_trans = torchaudio.transforms.Resample(self.sampling_rate, 16000)

        # Load Database  for audio embeddings
        try:
            self.db_embeddings_audio = EmbeddingsHandler(os.path.join(self.dataset_path, "audio"), n_neighbors=4)
            self.model_audio = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
        except FileNotFoundError:
            info(f"Unable to find dataset {EmbeddingsHandler(os.path.join(self.dataset_path, 'audio'))}")
            return False
        return True

    def getPeriod(self):
        """
           Module refresh rate.
           Returns : The period of the module in seconds.
        """
        return 0.05

    def record_audio(self):
        audio_file = AudioSegment.from_file("no_silence_audio_test.wav", "wav")
        chunk_length =1000 #pydub calculates in millisec
        chunks = make_chunks(audio_file, chunk_length) #make chunks of 1 sec

        for i, chunk in enumerate(chunks):
            chunk_name = "chunk{0}.wav".format(i)
            print("exporting", chunk_name)
            chunk.export(chunk_name, format="wav")

    def check_voice(self):
        if self.eventPort.getInputCount():
            event_name = self.eventPort.read(False)
            if event_name:
                event_name = event_name.get(0).asString()
                if event_name == "start_voice":
                    self.is_voice = True
                elif event_name == "stop_voice":
                    self.audio = []
                    self.nb_samples_received = 0
                    self.is_voice = False
                else:
                    pass

    def updateModule(self):
        speaker_name, audio_score = "unknown", 0
        print("Computing Speaker Embedding")
        audio_signal = self.format_signal(self.audio)
        # Compute speaker embeddings and do speaker prediction only if the audio database is updated with
        # the same people folders as the face embedding folders (make empty folders?)
        self.speaker_emb = self.get_audio_embeddings(audio_signal)

    def format_signal(self, audio_list_samples):
        """
        Format an audio given a list of samples
        :param audio_list_samples:
        :return: numpy array
        """
        np_audio = np.concatenate(audio_list_samples, axis=1)
        np_audio = np.squeeze(np_audio)
        signal = np.transpose(np_audio, (1, 0))
        signal = signal.mean(axis=1)

        return signal

    def get_audio_embeddings(self, audio):
        """
        Generate voice embedding from audio sample
        :param audio:
        :return:
        """
        resample_audio = self.resample_trans(torch.from_numpy(audio.transpose()))
        embedding = self.model_audio.encode_batch(resample_audio)
        embedding = embedding.squeeze(axis=0)

        return embedding

    def predict_speaker(self, embedding):

        score, speaker_name = self.db_embeddings_audio.get_speaker_db_scan(embedding)

        if score == -1:
            speaker_name = "unknown"

        self.db_embeddings_audio.excluded_entities = []
        print("Predicted speaker name is {} with score {}".format(speaker_name, score))

        return speaker_name, float(score)


if __name__ == '__main__':

   exit()