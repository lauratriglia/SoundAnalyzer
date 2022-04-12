import librosa
import soundfile as sf

audio_file = r'voiceRecognition/filtered1.wav'
# read wav data
audio, sr = librosa.load(audio_file, sr=8000, mono=True)
print(audio.shape, sr) #(837632,) 8000 -> this wav file contains 837632 length data

#to remove all silence in a wav file it can be used a librosa.effect.split() function

clips = librosa.effects.split(audio, top_db=10)
print(clips)

wav_data=[]
for c in clips:
    print(c)
    data=audio[c[0]: c[1]]
    wav_data.extend(data)

sf.write('silence_filtered.wav', wav_data, sr)
