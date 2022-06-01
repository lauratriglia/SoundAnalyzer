## Sound Analyzer
In the repository you will find some python scripts devoted to the manipulation of audio.wav. 
* __create_confusion.py__: Even if the name can be a little bit distracting, this python script is dedicated to create a confusion matrix, using a given database. In this case the databased used was DB_s that has all the audio collected and processed through __remove_voice.py__ and __emb_creation1.py__
* __emb_creation1.py__: this python script is dedicated to create embeddings from chunks of a second, which are also created in the same script.  make_chunk creates chunks of one second from the audio in the database.
Once each chunks is exported in the folder n_chunks, with a classifier from speechbrain (https://colab.research.google.com/drive/1UwisnAjr8nQF3UnrkIJ4abBMAWzVwBMh?usp=sharing),
        there is the embedding creation. Each embedding is stored in a list, which is saved as a .npy file.
* __make_dir.py__: little script to create a folder to contain the chunks create in __emb_creation1.py__.
* __prediction.py__: this script is dedicated to take the embeddings previously created with emb_creation1 and uses a KNN model to return the
    predicted speaker. The KNN model is in __speaker_embeddings.py__.
* __remove_voice.py__: it splits an audio in intervals in which there's only voice. To do that the librosa
    library is used, which splits the audio according to a threshold (in decibels) below reference to consider as
    silence
* __speaker_embeddings__: code of Jonas Gonzales


### How the database is constructed
To be able to work incrementally with the database, the database is divided in such way:
```bash
DB_folder
|___ Partecipant names
     |___ Dates
          |___ .wav file
```

