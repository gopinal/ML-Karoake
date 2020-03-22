
# coding: utf-8

# In[70]:


# Importing libraries we'll need to make spectrograms--all are available with Anaconda3 installation except soundfile
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio
# Used for importing sound files in wav format; installed in command prompt with: pip install soundfile
import soundfile as sf
from scipy.fftpack import fft
from scipy.io import wavfile
from scipy import signal
from numpy import array
from numpy import reshape  # needed for re-rolling vectors into matrices
import warnings


class Spectrogram:
    def __init__(self, audio_file, contains_vocals):
        # contains_vocals is a boolean that specifies whether the songs has vocals (true) or is a karaoke (false)
        self.contains_vocals = contains_vocals

        # Importing audio files, with PySoundFile: https://pysoundfile.readthedocs.io/en/0.8.1/
        # According to our tutorial, we want our samples to be sound files 500 ms long
        self.song_data, self.sample_rate = sf.read(audio_file)

        # sample_rate is an integer, denoting the sampling frequency in Hz of the song (i.e. the time resolution of the
        # song_data). song_data is an n-column array, where rows represent signal amplitude at a unit of time,
        # and each column represents a channel. For stereo songs there are two channels, hence two columns. For ease of
        # processing, we'll turn
        # stereo songs into mono songs:
        if self.song_data.shape[1] > 1:
            song_data = (self.song_data[:, 0] +
                         self.song_data[:, 1]) / 2  # Averages out each channel in stereo song into a mono channel
            # We want to segment our song files into individual examples of 500 ms; the following will
            # generate these examples:
        seconds = song_data.shape[0]/self.sample_rate #Total number of seconds in song
        self.n = int(seconds*2) #Number of examples we'll make, each 500 ms, from song data we loaded

        pxl_per_samp = round(song_data.shape[0]/(seconds*2)) #Pixels per sample--think of our data as an image
       
        song_array = np.zeros(shape=(self.n,pxl_per_samp))
             
        for i in range(0,(self.n-1)):
            song_array[i,:] = song_data[i*pxl_per_samp:(i+1)*pxl_per_samp]
            
        # Scipy methods for doing (inverse) short-time fourier transforms, or STFT,
        # (that is to reproduce our song from STFT)
        L = 1024
        noverlap = 4 #The hop-size is L/noverlap = 256

        self.f, self.t, _ = signal.stft(song_array[0, :], fs=self.sample_rate, nperseg=L, noverlap=noverlap)
        
        self.spec_exmpl_3D_array = np.zeros(shape=(self.n, len(self.f), len(self.t)))
        
        #The following loop builds a 3D array from the spectra generated for each 500 ms sample from the song 
        
        for i in range(0, (self.n - 1)):
            _, _, self.spec = signal.stft(song_array[i, :], fs=self.sample_rate, nperseg=L, noverlap=noverlap)
            self.spec_exmpl_3D_array[i,:,:] = self.spec
            
        self.spec_exmpl_3D_array = self.spec_exmpl_3D_array.reshape(self.spec_exmpl_3D_array.shape[0],self.spec_exmpl_3D_array.shape[1],self.spec_exmpl_3D_array.shape[2],-1)
        #spec is an array of dimensions (len(f), len(t)), the rows being along the frequency axis
        
        # f is a list containing the frequencies that will be plotted, size of 513 with L = 1024 and hop size = 256
        # t is a list containing the units of time where data will be plotted, size of 88 for 500 ms long samples
        
        
        ###The following lines are for unrolling the STFT into one row###
        #spec_exmpl_array = np.zeros(shape=(n, len(self.f) * len(self.t)))
        
        #for i in range(0, (n - 1)):
        #    _, _, self.spec = signal.stft(song_array[i, :], fs=self.sample_rate, nperseg=L, noverlap=noverlap)
        #    # spec is an array of dimensions (len(f), len(t)), the rows being along the frequency axis
        #    spec_exmpl_array[i, :] = self.spec.ravel()  # Unrolls spectrogram of example and puts it into array
        #####
        
        
        # The following is for appending a text file with the spec_exmpl_array data to build up a test set for our
        # model. It also includes y values for the corresponding examples by reading the file name of song we
        # imported to get this data
        
        #data_array = None
        #if self.contains_vocals:
            # If examples come from a song with vocals (y = 1)
        #    data_array = np.ones((spec_exmpl_array.shape[0], spec_exmpl_array.shape[1] + 1))
        #else:
            # If examples come from a song with no vocals (y = 0)
        #    data_array = np.zeros((spec_exmpl_array.shape[0], spec_exmpl_array.shape[1] + 1))

        #data_array[:, :-1] = spec_exmpl_array  # Now data_array is our X with the y column attached at the end
        #filename = 'data_matrix_500ms.txt'
        #with open(filename, mode='a') as data_file:
        #    np.savetxt(data_file, data_array)
        
    def get_X(self):
        # To feed into a neural network, we'd use:
        return (self.spec_exmpl_3D_array)
    
    def get_Y(self):
        # To feed into a neural network, we'd use:
        data_array = None
        if self.contains_vocals:
            # If examples come from a song with vocals (y = 1)
            self.Y = np.ones((self.n,1))
        else:
            # If examples come from a song with no vocals (y = 0)
            self.Y = np.zeros((self.n,1))  

        return (self.Y)   

        #Haven't tried the following functions since we've updated this class to make 3D arrays rather than unrolled 2D arrays

    def plot_spectrogram(self):
        plt.pcolormesh(self.t, self.f, np.abs(self.spec), vmin=0, vmax=abs(np.amax(self.spec)))  # cmap='gray')
        # Previous line sets the scale of intensity for heat map

        plt.ylim([self.f[1], self.f[-1]])
        plt.title('STFT Magnitude')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.yscale('symlog', linthreshy=300)
        plt.show()
        return

    # Plots the fourier transform of the whole song snippet
    def plot_fourier(self):
        x = self.song_data

        fourier_transform = np.fft.fft(x) / np.amax(x)  # Normalized?
        fourier_transform = fourier_transform[range(int(len(x) / 2))]  # Exclude sampling frequency

        tp_count = len(x)
        values = np.arange(int(tp_count / 2))
        time_period = tp_count / self.sample_rate
        frequencies = values / time_period

        plt.title('Fourier transform depicting the frequency components')
        plt.plot(frequencies, abs(fourier_transform))
        plt.xlabel('Frequency')
        plt.ylabel('Amplitude')
        plt.xscale('log')
        plt.show()
        return

    def plot_signal(self):
        ts = self.song_data
        total_ts_sec = len(ts) / self.sample_rate
        print("The total time series length = {} sec (N points = {}) ".format(total_ts_sec, len(ts)))
        plt.figure(figsize=(20, 3))
        plt.plot(ts)
        plt.xticks(np.arange(0, len(ts), self.sample_rate),
                   np.arange(0, len(ts) / self.sample_rate, 1))
        plt.ylabel("Amplitude")
        plt.xlabel("Time (second)")
        plt.title(
            "The total length of time series = {} sec, sample_rate = {}".format(len(ts) / self.sample_rate, self.sample_rate))
        plt.show()
        return



