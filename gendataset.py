
# coding: utf-8

# In[11]:


###How to use this script###

#Parameters:

#DIR = the directory containing your songs; should be either instrumentals or vocal+instruments from the test or training set
#contains_vocals = 1 if the songs have vocals, 0 if they're instrumentals and don't have vocals
#n_samples = 24505 for test set, 46200 for training set

#You need the latest spectrogram.py to use this script. For the variable DIR, you will specify the directory where your 
#music files are.

#The end product of the script is the two complete arrays of X and Y generated from all the songs in the specified folder.
#X will be a [n_samples x 513 x 23] 3D array
#Y will be an [n_samples] vector (specifically for Python, a list)


# In[2]:


#Python3 code to read multiple file names in a directory or folder to write 
#References: https://stackoverflow.com/questions/2632205/how-to-count-the-number-of-files-in-a-directory-using-python
#https://www.geeksforgeeks.org/rename-multiple-files-using-python/


# In[3]:


from spectrogram import Spectrogram #Importing the class Tuan made. You should have spectrogram.py in the same
                                                                                    #directory as this script
import os, os.path #Needed for reading file names directly from folder
import numpy as np


# In[10]:


#The directory where your songs are stored
DIR = "C:/Users/numbe/Documents/Academics/Coursework/CS 129 - Machine Learning/Final Project/musdb18/test/Instrumental Versions"

os.chdir(DIR) #Sets working directory to where song files are; this is so we can load them and write the dataset file in there

#The line below makes a list of all the filenames in the specified directory, while excluding names of subdirectories/folders
filename_list = [name for name in os.listdir('.') if os.path.isfile(name)]

print('Processing songs from:'+DIR)

#########################################
#For test set, n_samples = 24505
#For training set, n_samples = 46200
freq_size = 513
time_size = 23
n_samples = 24505
#These are dimensions of the matrix [X, Y]
#########################################

X = np.zeros((n_samples,freq_size,time_size))
Y = np.zeros((n_samples))
n_samples_so_far = 0 #This will ensure we are indexing correctly to put the x for our sample into our X matrix of all songs' data

for i in range(1,len(filename_list)):
        filename = filename_list[i]; #Iterates through each file name in the list
        
        print(filename)
        
        contains_vocals = 0; #If the songs are instrumental/karaoke, it should be 0; if it has vocals, value should be 1
        spect = Spectrogram(filename,contains_vocals) #Calling the Spectrogram class. This is what creates/updates our dataset textfile
        x_is = spect.get_X() #numpy array data type
        y_is = spect.get_Y() #numpy array data type
        
        n_song_samples = np.shape(y_is)[0] #I.e., how many samples did we make from the song (song length (sec)*2, since 500ms segments used)
        
        upper_index = n_samples_so_far + n_song_samples #Each sample is a row in our array/matrix
        
        X[n_samples_so_far:upper_index,:,:] = x_is #Puts in values of samples from song into the matrix [X,Y] of all samples
        Y[n_samples_so_far:upper_index] = y_is #^^^
        
        n_samples_so_far = n_samples_so_far + n_song_samples #Updates how many samples we've done so far

