# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 16:59:44 2018

@author: Rogier
"""
import audacity
channel=0

fp = open("C:\\Users\\Rogier\\Documents\\Python_Scripts\\new6.txt")
for line in fp:
    print(line)
    q=line.find("aup")
    lstr=line[0:q+3]
    aup = audacity.Aup(lstr)
    ff=lstr[:-4]
    ff=ff + ".wav"
    aup.towav(ff, channel, start=0, stop=None)
fp.close()


                      
