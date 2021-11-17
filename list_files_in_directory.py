# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 15:06:35 2018

@author: Rogier
"""

import os
from fnmatch import fnmatch

root = 'E:\Social Interaction Test Videos\Habituation\selected_for_tracking'
pattern = "*.mp4"

for path, subdirs, files in os.walk(root):
    for name in files:
        if fnmatch(name, pattern):
#            print (os.path.join(path, name))
            print (name)            