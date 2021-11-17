# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 22:17:55 2018

@author: Rogier
"""

import glob, os, shutil

#from os.path import isfile, join
#dir1=listdir("E:\\Dropbox (MIT)\\Marmoset labeled audio")
round1='Mixed_order\\'
source_dir='E:\\Dropbox (MIT)\\Baby macaques\\Eyetracking\\' + round1
dest_dir='J:\No_internet_analysis\\' + round1
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

files = glob.iglob(os.path.join(source_dir, "*.tsv"))
for file in files:
    if os.path.isfile(file):
        print('\n copying ' + file)
        shutil.copy2(file, dest_dir)