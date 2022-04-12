import librosa
import soundfile as sf

audio_file = r'samples_audio/audio_test.wav'
audio_file1 = r'samples_audio/audio_test1.wav'
audio_file2 = r'samples_audio/audio_test2.wav'
# read wav data
audio, sr = librosa.load(audio_file, sr=8000, mono=True)
audio1, sr = librosa.load(audio_file1, sr=8000, mono=True)
audio2, sr = librosa.load(audio_file2, sr=8000, mono=True)

# to remove all silence in a wav file it can be used a librosa.effect.split() function

clips = librosa.effects.split(audio, top_db=7)
clips1 = librosa.effects.split(audio1, top_db=7)
clips2= librosa.effects.split(audio2, top_db=7)


wav_data=[]
wav_data1=[]
wav_data2=[]
for c in clips:
    data=audio[c[0]: c[1]]
    wav_data.extend(data)
for c in clips1:
    data = audio[c[0]: c[1]]
    wav_data1.extend(data)
for c in clips2:
    data=audio[c[0]: c[1]]
    wav_data2.extend(data)

sf.write('samples_audio/audio_sample.wav', wav_data, sr)
sf.write('samples_audio/audio_sample1.wav', wav_data1, sr)
sf.write('samples_audio/audio_sample2.wav', wav_data2, sr)