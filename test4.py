import numpy as np

from scipy import fftpack

import pyaudio
import wave

from scipy.io import wavfile


def playback():
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    RECORD_SECONDS = 8
    WAVE_OUTPUT_FILENAME = "output.wav"

    filename = 'no_silence_audio_test.wav'

    # Set chunk size of 1024 samples per data frame
    chunk = 1024

    # Open the sound file
    wf = wave.open(filename, 'rb')

    frame_rate = wf.getframerate()

    wf_x = wf.readframes(-1)

    signal = np.frombuffer(wf_x, dtype='int16')

    # print("signalxx", signal)

    return [signal, frame_rate]


time_step = 0.5

# get the data
data = playback()

sig = data[0]
frame_rate = data[1]

# Return discrete Fourier transform of real or complex sequence
sig_fft = fftpack.fft(sig)  # tranform the sin function

# Get Amplitude ?
Amplitude = np.abs(sig_fft)  # np.abs() - calculate absolute value from a complex number a + ib

Power = Amplitude ** 2  # create a power spectrum by power of 2 of amplitude

# Get the (angle) base spectrum of these transform values i.e. sig_fft
Angle = np.angle(sig_fft)  # Return the angle of the complex argument

# For each Amplitude and Power (of each element in the array?) - there is will be a corresponding difference in xxx

# This is will return the sampling frequecy or corresponding frequency of each of the (magnitude) i.e. Power
sample_freq = fftpack.fftfreq(sig.size, d=time_step)

print(Amplitude)
print(sample_freq)

# Because we would like to remove the noise we are concerned with peak freqence that contains the peak amplitude
Amp_Freq = np.array([Amplitude, sample_freq])

# Now we try to find the peak amplitude - so we try to extract
Amp_position = Amp_Freq[0, :].argmax()

peak_freq = Amp_Freq[1, Amp_position]  # find the positions of max value position (Amplitude)

# print the position of max Amplitude
print("--", Amp_position)
# print the frequecies of those max amplitude
print(peak_freq)

high_freq_fft = sig_fft.copy()
# assign all the value the corresponding frequecies larger than the peak frequence - assign em 0 - cancel!! in the array (elements) (?)
high_freq_fft[np.abs(sample_freq) > peak_freq] = 0

print("yes:", high_freq_fft)

# Return discrete inverse Fourier transform of real or complex sequence
filtered_sig = fftpack.ifft(high_freq_fft)

# Using Fast Fourier Transform and inverse Fast Fourier Transform we can remove the noise from the frequency domain (that would be otherwise impossible to do in Time Domain) - done.
print("filtered noise: ", filtered_sig)

print("getiing frame rate $$", frame_rate)

filteredwrite = np.fft.irfft(filtered_sig, axis=0)

print(filteredwrite)

wavfile.write('TestFiltered.wav', frame_rate, filteredwrite)
