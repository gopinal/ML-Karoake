
# coding: utf-8

# In[ ]:


###How to use this script###

#You need spectrogram.py to use this script. For the variable DIR, you will specify the directory where your #music files are.
#The end product of the script is a text file containing an array/matrix [X Y] of unrolled STFTs of 500ms segments of these songs. 


# In[2]:


from spectrogram import Spectrogram #Importing the class Tuan made. You should have spectrogram.py in the same
                                                                                    #directory as this script


# In[3]:


#Python3 code to read multiple file names in a directory or folder to write 
#References: https://stackoverflow.com/questions/2632205/how-to-count-the-number-of-files-in-a-directory-using-python
#https://www.geeksforgeeks.org/rename-multiple-files-using-python/

# importing os module 
import os, os.path

DIR = "C:/Users/.../musdb18/test/Instrumental Versions" #The directory where your songs are stored
os.chdir(DIR) #Sets working directory to where song files are; this is so we can load them and write the dataset file in there

#The line below makes a list of all the filenames in the specified directory, while excluding names of subdirectories/folders
filename_list = [name for name in os.listdir('.') if os.path.isfile(name)]



# In[4]:


print('Processing songs from:'+DIR)
os.chdir(DIR)

for i in range(1,len(filename_list)):
        filename = filename_list[i]; #Iterates through each file name in the list
        print(filename)
        contains_vocals = 0; #If the songs are instrumental/karaoke, it should be 0; if it has vocals, value should be 1
        Spectrogram(filename,0) #Calling the Spectrogram class. This is what creates/updates our dataset textfile


# In[5]:


filename_list

