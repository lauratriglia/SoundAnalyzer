import librosa
import soundfile as sf
import os


def remove_silence(audio_file, counter):
    dir_name = os.path.dirname(audio_file)
    print(dir_name)
    #if '.wav' in dir_name:
        # read wav data
    audio, sr = librosa.load(audio_file, sr=8000, mono=True)
    # to remove all silence in a wav file it can be used a librosa.effect.split() function
    clips = librosa.effects.split(audio, top_db=10)
    wav_data=[]
   # for c in clips:
   #     data=audio[c[0]: c[1]]
   #     wav_data.extend(data)
    for i in range(len(clips)-1):
        start_silence = clips[i][1]
        end_silence = clips[i+1][0]
        silence = audio[start_silence: end_silence]
        wav_data.extend(silence)

    sf.write('DB_s/Silence_audio' + '/' + "silence{0}.wav".format(counter), wav_data, sr)
    # counter += 1


main_dir = 'DB_raw'
all_file_names = os.listdir(main_dir)
counter = 1

for subdir in all_file_names:
    if os.path.isdir(main_dir + '/' + subdir):
        dir_names = os.listdir(main_dir + '/' + subdir)
        print(subdir)
        files = os.listdir(main_dir + '/' + subdir)
        for subdir1 in dir_names:
            if os.path.isdir(main_dir+'/'+subdir+'/'+subdir1):
                files = os.listdir(main_dir+'/'+subdir+'/'+subdir1)
                for file in files:
                     if '.wav' in file:
                      remove_silence(main_dir+'/'+subdir+'/'+subdir1+'/'+file, counter)
                      counter += 1