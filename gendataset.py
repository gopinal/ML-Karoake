# The end product of the script is the two complete arrays of X and Y generated from all the songs in
# the specified folder.
# X will be a [n_samples x 513 x 23] 3D array
# Y will be an [n_samples] vector (specifically for Python, a list)

# Python3 code to read multiple file names in a directory or folder to write
# References: https://stackoverflow.com/questions/2632205/how-to-count-the-number-of-files-in-a-directory-using-python
# https://www.geeksforgeeks.org/rename-multiple-files-using-python/

from spectrogram import Spectrogram
import os, os.path
import numpy as np


# user_dir is the directory where your songs are stored
# contains_vocals is whether it is a directory of karaoke songs (false) or a directory of full songs (true)
# user_dir = "C:/Users/numbe/Documents/Academics/Coursework/CS 129 - " \
#      "Machine Learning/Final Project/musdb18/test/Instrumental Versions"#
class Gendataset:
    def __init__(self, user_dir, contains_vocals):
        os.chdir(user_dir)
        # The line below makes a list of all the filenames in the specified directory,
        # while excluding names of subdirectories/folders
        self.filename_list = [name for name in os.listdir('.') if os.path.isfile(name)]
        print('Processing songs from:' + user_dir)
        # For test set, n_samples = 24505
        # For training set, n_samples = 46200
        freq_size = 513
        time_size = 23
        n_samples = 24505
        # These are dimensions of the matrix [X, Y]

        self.X = np.zeros((n_samples, freq_size, time_size), dtype=int)
        self.Y = np.zeros(n_samples, dtype=int)
        # This will ensure we are indexing correctly to put the x for our sample into our X matrix of all songs' data
        n_samples_so_far = 0
        for i in range(len(self.filename_list)):
            filename = self.filename_list[i]  # Iterates through each file name in the list
            print(filename)

            self.contains_vocals = contains_vocals
            spect = Spectrogram(filename, contains_vocals)
            x_is = spect.get_x()
            y_is = spect.get_y()

            # I.e., how many samples did we make from the song (song length (sec)*2, since 500ms segments used)
            n_song_samples = np.shape(y_is)[0]

            upper_index = n_samples_so_far + n_song_samples  # Each sample is a row in our array/matrix

            self.X[n_samples_so_far:upper_index, :, :] = x_is
            self.Y[n_samples_so_far:upper_index] = y_is

            n_samples_so_far = n_samples_so_far + n_song_samples  # Updates how many samples we've done so far
