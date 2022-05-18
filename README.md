## Sound Analyzer
In the repository you will find some python scripts devoted to the manipulation of audio.wav. 
* __test.py__: remove silence from an .wav file, using librosa library.
* __test2.py__: create chunks of the .wav audio and extract the embeddings of each chunk.
* __test3.py__: extract embeddings directly from the .wav file. 
* __embeddings_creation__: each .wav file that is contained in the database is processed, divided in chunks and from each of them it is extracted the embeddings. Each embedding is saved in the specific folder
### How the database is constructed
To be able to work incrementally with the database, the database is divided in such way:
```bash
DB_folder
|___ Partecipant names
     |___ Dates
          |___ .wav file
```

