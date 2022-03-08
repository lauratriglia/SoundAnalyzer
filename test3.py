import torchaudio
from speechbrain.pretrained import EncoderClassifier
# create the embeddings of the audio (that is not divided in chunks)
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
signal, fs = torchaudio.load('no_silence_audio_test.wav')

embeddings = classifier.encode_batch(signal)

output_probs, score, index, text_lab = classifier.classify_batch(signal)

print(output_probs)
print(score)
print(index)
print(text_lab)
print("Predicted: " + text_lab[0])
print(embeddings)