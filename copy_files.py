# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 22:17:55 2018

@author: Rogier
"""

import shutil
from os import listdir
from os import walk
from os import path
from os import makedirs

#from os.path import isfile, join
#dir1=listdir("E:\\Dropbox (MIT)\\Marmoset labeled audio")
lookdir="E:\\Dropbox (MIT)\\Marmoset labeled audio\\"
dstdir="E:\\Dropbox (MIT)\\Zeyun Wu\\labels_txt\\"
#if path.exists(dstdir):
#    shutil.rmtree(dstdir) 
#if not os.path.exists(dstdir):
#    os.makedirs(dstdir)
#makedirs(dstdir)
    
dirs=next(walk(lookdir))[1]
L=len(dirs)
for c in range(0,L):
   print(c) 
   arr_txt = [x for x in listdir(lookdir+dirs[c]) if x.endswith(".txt")]
   LL=len(arr_txt)
   for d in range(0,LL):
       if ('Condition' not in arr_txt[d]) & ('condition' not in arr_txt[d]) & ('Other' not in arr_txt[d]):    
           src=lookdir+dirs[c]+'\\'+arr_txt[d]
           dst=dstdir+dirs[c][0:8]+'_'+arr_txt[d]
           shutil.copyfile(src, dst)
       


