# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 22:17:55 2018

@author: Rogier
"""



import glob, os

import os
path = 'E:\\Dropbox (MIT)\\Marmoset_call_dictionary_room_sim\\tr'
files = os.listdir(path)
i = 1

for file in files:
    os.rename(os.path.join(path, file), os.path.join(path, 'room_tr'+str(i)+'.wav'))
    i = i+1
		
		
path = 'E:\\Dropbox (MIT)\\Marmoset_call_dictionary_room_sim\\ts'
files = os.listdir(path)
i = 1

for file in files:
    os.rename(os.path.join(path, file), os.path.join(path, 'room_ts'+str(i)+'.wav'))
    i = i+1		

path = 'E:\\Dropbox (MIT)\\Marmoset_call_dictionary_room_sim\\tw'
files = os.listdir(path)
i = 1

for file in files:
    os.rename(os.path.join(path, file), os.path.join(path, 'room_tw'+str(i)+'.wav'))
    i = i+1