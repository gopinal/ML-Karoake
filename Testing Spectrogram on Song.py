
# coding: utf-8

# In[61]:


#Using reference from: https://fairyonice.github.io/implement-the-spectrogram-from-scratch-in-python.html#Create-Spectrogram
#to do a STFT of two short sound clips from a song, one being the karaoke version


# In[145]:


#Importing libraries we'll need to make spectrograms
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio
import soundfile as sf #installed in command prompt with: pip install soundfile
from scipy.fftpack import fft
from scipy.io import wavfile
from scipy import signal
from numpy import array 
from numpy import reshape #needed for re-rolling vectors into matrices
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 


# In[64]:


#Importing audio files, see with PySoundFile: https://pysoundfile.readthedocs.io/en/0.8.1/

song_data, sample_rate = sf.read('k_of_cyd_0p6s.wav')

kk_song_data, sample_rate = sf.read('k_of_cyd_0p6s_karaoke.wav')

#song_data is a two-column matrix, where rows represent signal amplitude at a unit of time, and each column represents a channel
#sample_rate is an integer, denoting the sampling frequency in Hz of the song (i.e. the time resolution of the song_data)


# In[69]:


#To play song files, use this function
Audio('k_of_cyd_0p6s.wav', rate=sample_rate)


# In[70]:


#Now the karaoke version
Audio('k_of_cyd_0p6s_karaoke.wav', rate=sample_rate)


# In[71]:


#Plotting the waveform of the original song with vocals; note since it has two channels we can plot them separately
ts = song_data[:,0]
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

ts = song_data[:,1]
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


# In[72]:


#Plotting the waveform of the karaoke song; though it also has two channels we can plot them together and see that they differ
#This is likely because the removed vocal components dominated in one channel
ts = kk_song_data
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


# In[73]:


from IPython.display import Image
print('The following functions will calculate the Fourier coefficients for the Fourier series representation of our sound file. Recall that any signal can be arbitrarily represented as an infinite sum of sinusoids with some weighting factor, or fourier coefficients, expressed in this first image in the exponential notation:')
Image('fourierseries.png')


# In[74]:


print('Each of these fourier coefficients can be calculated after using some clever algebra and calculus with the expression:')
Image('fouriercoefficients.png')


# In[75]:


def get_xn(Xs,n):
    '''
    calculate the Fourier coefficient X_n of 
    Discrete Fourier Transform (DFT)
    '''
    L  = len(Xs)
    ks = np.arange(0,L,1)
    xn = np.sum(Xs*np.exp((1j*2*np.pi*ks*n)/L))/L
    return(xn)

def get_xns(ts):
    '''
    Compute Fourier coefficients only up to the Nyquest Limit Xn, n=1,...,L/2
    and multiply the absolute value of the Fourier coefficients by 2, 
    to account for the symetry of the Fourier coefficients above the Nyquest Limit. 
    '''
    mag = []
    L = len(ts)
    for n in range(int(L/2)): # Nyquest Limit
        mag.append(np.abs(get_xn(ts,n))*2)
    return(mag)


# In[76]:


mag = get_xns(list(song_data[:,0])) #Note that the fourier coefficient function won't work if we input the signal as a matrix


# In[354]:



# the number of points to label along xaxis
Nxlim = 10

plt.figure(figsize=(20,3))
plt.plot(mag)
plt.xlabel("Frequency (k)")
plt.title("Two-sided frequency plot")
plt.ylabel("|Fourier Coefficient|")
plt.xscale("log")
plt.show()


# In[353]:


def get_Hz_scale_vec(ks,sample_rate,Npoints):
    freq_Hz = ks*sample_rate/Npoints
    freq_Hz  = [int(i) for i in freq_Hz ] 
    return(freq_Hz )

ks   = np.linspace(0,len(mag),Nxlim)
ksHz = get_Hz_scale_vec(ks,sample_rate,len(ts))

plt.figure(figsize=(20,3))
plt.plot(mag)
plt.xticks(ks,ksHz)
plt.title("Frequency Domain")
plt.xlabel("Frequency (Hz)")
plt.ylabel("|Fourier Coefficient|")
plt.xscale("log")
plt.show()


# In[84]:


def create_spectrogram(ts,NFFT,noverlap = None):
    '''
          ts: original time series
        NFFT: The number of data points used in each block for the DFT.
          Fs: the number of points sampled per second, so called sample_rate
    noverlap: The number of points of overlap between blocks. The default value is 128. 
    '''
    if noverlap is None:
        noverlap = NFFT/2
    noverlap = int(noverlap)
    starts  = np.arange(0,len(ts),NFFT-noverlap,dtype=int)
    # remove any window with less than NFFT sample size
    starts  = starts[starts + NFFT < len(ts)]
    xns = []
    for start in starts:
        # short term discrete fourier transform
        ts_window = get_xns(ts[start:start + NFFT]) 
        xns.append(ts_window)
    specX = np.array(xns).T
    # rescale the absolute value of the spectrogram as rescaling is standard
    spec = 10*np.log10(specX)
    assert spec.shape[1] == len(starts) 
    return(starts,spec)


# In[384]:


def plot_spectrogram(spec,ks,sample_rate, L, starts, mappable = None):
    plt.figure(figsize=(20,8))
    plt_spec = plt.imshow(spec,origin='lower') #,cmap='gray')

    ## create ylim
    Nyticks = 10
    ks      = np.linspace(0,spec.shape[0],Nyticks)
    ksHz    = get_Hz_scale_vec(ks,sample_rate,len(ts))
    plt.yticks(ks,ksHz)
    plt.ylabel("Frequency (Hz)")
    
    ## create xlim
    Nxticks = 2
    ts_spec = np.linspace(0,spec.shape[1],Nxticks)
    ts_spec_sec  = ["{:4.2f}".format(i) for i in np.linspace(0,total_ts_sec*starts[-1]/len(ts),Nxticks)]
    plt.xticks(ts_spec,ts_spec_sec)
    plt.xlabel("Time (sec)")

    plt.title("Spectrogram L={} Spectrogram.shape={}".format(L,spec.shape))
    plt.colorbar(mappable,use_gridspec=True)
    plt.show()
    return(plt_spec)


# In[385]:


L = 1024 #Window size
noverlap = 256 #hop size
starts, spec = create_spectrogram(list(song_data[:,0]),L,noverlap = noverlap ) #Note that the fourier coefficient function won't
#work if we input the signal, song_data, as a matrix--we must also pick one channel
#Spec is a 512x34 matrix, where each row represents intensity at some frequency, and each column represents a slice in time
plot_spectrogram(spec,ks,sample_rate,L, starts)


# In[169]:


#Now let's try it with the karoake song
starts, spec = create_spectrogram(list(kk_song_data[:,0]),L,noverlap = noverlap ) #Note that the fourier coefficient function won't
#work if we input the signal, song_data, as a matrix--we must also pick one channel
plot_spectrogram(spec,ks,sample_rate,L, starts)


# In[123]:


#Methods needed to unroll and reshape matrices--might need to make inputs into model

spec_row_vector = spec.ravel() #Unrolls the spectrogram matrix column-by-column (forward in time) and then down by
# rows (lower and lower frequencies)
len(spec_row_vector)
new_spec = spec_row_vector.reshape(spec.shape[0],spec.shape[1]) #The .shape method works like size() in octave or matlab


# In[285]:


#Scipy methods for doing (inverse) short-time fourier transforms, or STFT, (that is to reproduce our song from STFT)
#References:
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.stft.html#scipy.signal.stft
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.istft.html#scipy.signal.istft

L= 1024
#L = 512 #Smaller window size L gives us better time resolution but less frequency resolution due to the uncertainty principle
f, t, alt_spec = signal.stft(song_data[:,0], fs=sample_rate, nperseg=L, noverlap = 3*L/4) #The tutorial uses a hop size of 256,
#or L/4, which means the number of points that will overlap each time we take a slice for STFT is 3*L/4
_, new_song_data = signal.istft(alt_spec, fs=sample_rate, nperseg=L, noverlap = 3*L/4)


# In[377]:


plt.figure()
plt.pcolormesh(t, f, np.abs(alt_spec), vmin=0, vmax=abs(np.amax(alt_spec))) #cmap='gray') #Sets the scale of intensity for heat map
plt.ylim([f[1], f[-1]])
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.yscale('symlog', linthreshy=300)
plt.show()


# In[286]:


#To play song files, use this function
Audio(new_song_data, rate=sample_rate)


# In[378]:


f, t, kk_alt_spec = signal.stft(kk_song_data[:,0], fs=sample_rate, nperseg=L, noverlap = 3*L/4)

plt.figure()
vmax = 1 #maximum amplitude of time signal
plt.pcolormesh(t, f, np.abs(kk_alt_spec), vmin=0, vmax=abs(np.amax(kk_alt_spec))) #cmap='gray') #Sets the scale of intensity for heat map
plt.ylim([f[1], f[-1]])
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.yscale('symlog', linthreshy=300)
plt.show()


# In[379]:


subt_spec = alt_spec - kk_alt_spec #need to normalize?

plt.figure()
plt.pcolormesh(t, f, np.abs(subt_spec), vmin=0, vmax=abs(np.amax(subt_spec))) #cmap='gray') #Sets the scale of intensity for heat map
plt.ylim([f[1], f[-1]])
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.yscale('symlog', linthreshy=300)
plt.show()

