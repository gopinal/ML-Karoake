
# coding: utf-8

# In[131]:


#To understand more in detail how STFT works, refer to: https://fairyonice.github.io/implement-the-spectrogram-from-scratch-in-python.html#Create-Spectrogram
#Many of the parameters we use for the STFT and iSTFT are based on this tutorial: https://towardsdatascience.com/audio-ai-isolating-vocals-from-stereo-music-using-convolutional-neural-networks-210532383785


# In[132]:


#Importing libraries we'll need to make spectrograms--all are available with Anaconda3 installation except soundfile
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio
import soundfile as sf #used for importing sound files in wav format; installed in command prompt with: pip install soundfile
from scipy.fftpack import fft
from scipy.io import wavfile
from scipy import signal
from numpy import array 
from numpy import reshape #needed for re-rolling vectors into matrices
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


# In[133]:


#Importing audio files, with PySoundFile: https://pysoundfile.readthedocs.io/en/0.8.1/

#According to our tutorial, we want our samples to be sound files 500 ms long

song_data, sample_rate = sf.read('k_of_cyd_7s.wav')

#sample_rate is an integer, denoting the sampling frequency in Hz of the song (i.e. the time resolution of the song_data)
#song_data is an n-column array, where rows represent signal amplitude at a unit of time, and each column represents a channel
#For stereo songs there are two channels, hence two columns. For ease of processing, we'll turn stereo songs into mono songs:

if song_data.shape[1] > 1:
    song_data = (song_data[:,0]+song_data[:,1])/2 #Averages out each channel in stereo song into a mono channel


# In[134]:


#We want to segment our song files into individual examples of 500 ms; the following will generate these examples:
seconds = song_data.shape[0]/sample_rate #Total number of seconds in song
n = int(seconds*2) #Number of examples we'll make, each 500 ms, from song we loaded

pxl_per_samp = round(song_data.shape[0]/(seconds*2)) #Pixels per sample--think of our data as an image

song_array = np.zeros(shape=(n,pxl_per_samp))
             
for i in range(0,(n-1)):
    song_array[i,:] = song_data[i*pxl_per_samp:(i+1)*pxl_per_samp]

#Check whether the array has the right dimensions:
#x.shape


# In[135]:


#Scipy methods for doing (inverse) short-time fourier transforms, or STFT, (that is to reproduce our song from STFT)
#References:
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.stft.html#scipy.signal.stft
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.istft.html#scipy.signal.istft
 
L = 1024 #Window size, smaller sizes give us better time resolution but less frequency resolution due to the uncertainty principle
noverlap = 3*L/4 #The tutorial uses a hop size of 256, or L/4, which means the the number of points that will overlap each time
#we take a slice for STFT is 3*L/4

f, t, _ = signal.stft(song_array[0,:], fs=sample_rate, nperseg=L, noverlap=noverlap)
#f is a list containing all the frequencies that will be plotted, size of 513 with L = 1024 and hop size = 256
#t is a lost containing all the units of time where data will be plotted, size of 88 for 500 ms long samples

spec_exmpl_array = np.zeros(shape=(n,len(f)*len(t)))

for i in range(0,(n-1)):
    _, _, spec = signal.stft(song_array[i,:], fs=sample_rate, nperseg=L, noverlap=noverlap)
    #spec is an array of dimensions (len(f), len(t)), the rows being along the frequency axis and the columns along time
    spec_exmpl_array[i,:] = spec.ravel() #Unrolls spectrogram of example and puts it into array

#Check the size of spec__exmpl_array, where each row is an example, with:
#spec_exmpl_array.shape


# In[136]:


###The following code is for appending a text file with the spec_exmpl_array data to build up a test set for our model.###
#It also includes y values for the corresponding examples by reading the file name of song we imported to get this data.

#If examples come from a song with vocals (y = 1)
data_array = np.ones((spec_exmpl_array.shape[0], spec_exmpl_array.shape[1]+1))
#If examples come from a song with no vocals (y = 0)
data_array = np.zeros((spec_exmpl_array.shape[0], spec_exmpl_array.shape[1]+1))
data_array[:,:-1] = spec_exmpl_array #Now data_array is our X with the y column attached at the end

filename = 'data_matrix_500ms.txt'

with open(filename, mode='a') as data_file: #Mode 'a' is for appending, if it's 'w' it will overwrite data already in file
    np.savetxt(data_file,data_array)

#Ways to check how quickly code runs:

#%timeit b = np.hstack((a,np.zeros((a.shape[0],1))))
#10000 loops, best of 3: 19.6 us per loop

#%timeit b = np.zeros((a.shape[0],a.shape[1]+1)); b[:,:-1] = a
#100000 loops, best of 3: 5.62 us per loop


# In[137]:


# Read the array we stored in the textfile
with open(filename, mode='r') as data_file: #Mode 'r' is for reading
    loaded_data_array = np.loadtxt(data_file)

#To feed into a neural network, we'd use:
X = loaded_data_array[:,:-1]
y = loaded_data_array[:,loaded_data_array.shape[1]-1]
    
# Check whether the array we loaded is the same as the one we stored, otherwise following line returns an error
#assert np.all(new_data_array == data_array)
#You'll get an error if you appended to an existing data file rather than a new one.


# In[147]:


#This code reconstructs one example's SFTF array given the unrolled list, e.g. after they've been processed by our model

spec_list_example = spec_exmpl_array[0,:]

##To reconstruct our unrolled arrays of the spectrograms, use the following line:
rebuilt_spec = spec_list_example.reshape(spec.shape[0],spec.shape[1]) #The .shape method works like size() in octave or matlab

#After processing our data, we have to inverse fourier transform it to get the song back out
_, re_song_data = signal.istft(rebuilt_spec, fs=sample_rate, nperseg=L, noverlap=noverlap)


# In[148]:


#If all of that worked, then you will get out a song snippet of 500 ms:
Audio(re_song_data, rate=sample_rate)


# In[149]:


#There are different ways to visualize or listen to our data:
#To play song files, use this function

Audio(song_data, rate=sample_rate)


# In[150]:


#The following functions are for visualizing our songs as the amplitude over time, their fourier transforms, and spectrograms


# In[151]:


#You can visualize the signals and their respective STFTs with the following functions:

def plot_spectrogram(f, t, spec, mappable = None):
    plt.pcolormesh(t, f, np.abs(spec), vmin=0, vmax=abs(np.amax(spec))) #cmap='gray') 
    #Previous line sets the scale of intensity for heat map
  
    plt.ylim([f[1], f[-1]])
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.yscale('symlog', linthreshy=300)
    plt.show()

    return


# In[152]:


#Plots the fourier transform of the whole song snippet

def plot_fourier(song_data):

    x = song_data

    fourierTransform = np.fft.fft(x)/np.amax(x) # Normalized?
    fourierTransform = fourierTransform[range(int(len(x)/2))] # Exclude sampling frequency

    tpCount     = len(x)
    values      = np.arange(int(tpCount/2))
    timePeriod  = tpCount/sample_rate
    frequencies = values/timePeriod

    plt.title('Fourier transform depicting the frequency components')
    plt.plot(frequencies, abs(fourierTransform))
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    plt.xscale('log')
    plt.show()
    
    return


# In[153]:


def plot_signal(song_data):

    ts = song_data
    total_ts_sec = len(ts)/sample_rate
    print("The total time series length = {} sec (N points = {}) ".format(total_ts_sec, len(ts)))
    plt.figure(figsize=(20,3))
    plt.plot(ts)
    plt.xticks(np.arange(0,len(ts),sample_rate),
               np.arange(0,len(ts)/sample_rate,1))
    plt.ylabel("Amplitude")
    plt.xlabel("Time (second)")
    plt.title("The total length of time series = {} sec, sample_rate = {}".format(len(ts)/sample_rate, sample_rate))
    plt.show()

    return


# In[154]:


plot_signal(song_data)


# In[155]:


#Fourier transform of the whole signal

plot_fourier(song_data)


# In[156]:


plot_spectrogram(f,t,spec) #Spectrogram for the original song 

