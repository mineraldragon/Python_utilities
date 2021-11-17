# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 13:32:13 2018

@author: Rogier
"""

#read text into dataframe
import pandas as pd
data=pd.read_csv('E:\\Dropbox (MIT)\\Lucy\\Data\\20160614_Cricket.txt',sep='\t',header=None)
data.shape #prints (373,3) , number of rows and columns