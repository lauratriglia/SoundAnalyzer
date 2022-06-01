import os
import numpy as np
from pydub import AudioSegment
from pydub.utils import make_chunks
import librosa
import torchaudio
from speechbrain.pretrained import EncoderClassifier


def make_chunk(audio_file_name):
    '''
        :param str audio_file_name: corresponding to the location in which your audio file is stored.
        :return: nothing

        make_chunk creates chunks of one second from the audio in the database.
        Once each chunks is exported in the folder n_chunks, with a classifier from speechbrain
        (https://colab.research.google.com/drive/1UwisnAjr8nQF3UnrkIJ4abBMAWzVwBMh?usp=sharing),
        there is the embedding creation. Each embedding is stored in a list, which is saved as
        a .npy file.


    '''
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

    # check on the duration of the audio_file: in case is minor of 1 sec, it is not considered.
    if librosa.get_duration(filename=audio_file_name) < chunk_length/1000:
        return

    audio_file = AudioSegment.from_file(audio_file_name, "wav")
    chunks = make_chunks(audio_file, chunk_length)  # make chunks of 1 sec
    chunks_dir = os.path.join(dir_name, 'n_chunks')
    # print(chunks_dir)

    for i, chunk in enumerate(chunks):
        chunk_name = chunks_dir + '/' + "chunk{0}.wav".format(i)
        chunk.export(chunk_name, format="wav")
        # the chunks from which the embeddings are created, are only the ones which duration is greater than 1 sec
        if librosa.get_duration(filename= chunk_name) >= chunk_length/1000:
            signal, fs = torchaudio.load(chunk_name)
            embeddings = classifier.encode_batch(signal) # creation of the embeddings
            embeddings = embeddings.squeeze()
            list_emb.append(embeddings.numpy())

    with open(dir_name+'/chunk.npy', 'wb') as f:
        np.save(f, np.array(list_emb))


def main():
    main_dir = 'DB_Final'
    all_file_names = os.listdir(main_dir)
    # the structure of the database is Name_folder->Data_folders->.wav file
    for subdir in all_file_names:
        if os.path.isdir(main_dir + '/' + subdir):
            dir_names = os.listdir(main_dir + '/' + subdir)
        for subdir1 in dir_names:
            if os.path.isdir(main_dir+'/'+subdir+'/'+subdir1):
                files = os.listdir(main_dir+'/'+subdir+'/'+subdir1)
                for file in files:
                    if '.wav' in file:
                        make_chunk(main_dir+'/'+subdir+'/'+subdir1+'/'+file)


if __name__ == '__main__':
    main()