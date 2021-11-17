# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 23:10:29 2017
This plots fraction watched for submissive, neutral and threat clips, for mutants and wildtype
@author: Rogier
"""

import pandas as pd
import numpy as np
#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
#import h5py
from scipy import stats
import time

teller=-1
elapsed=[]
IDS=['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'M1', 'M2', 'M3', 'M4', 'M5']
#IDS=['C1']

df_fractwatched_1=pd.DataFrame()
df_fractwatched_2=pd.DataFrame()
df_fractwatched_3=pd.DataFrame()

df_pupresult_1_ag=pd.DataFrame()
df_pupresult_1_ne=pd.DataFrame()
df_pupresult_1_su=pd.DataFrame()

df_pupresult_2_ag=pd.DataFrame()
df_pupresult_2_ne=pd.DataFrame()
df_pupresult_2_su=pd.DataFrame()

df_pupresult_3_ag=pd.DataFrame()
df_pupresult_3_ne=pd.DataFrame()
df_pupresult_3_su=pd.DataFrame()

df_cumsum1=pd.DataFrame()
df_cumsum2=pd.DataFrame()
df_cumsum3=pd.DataFrame()
df_meanpup1=pd.DataFrame()
df_meanpup2=pd.DataFrame()
df_meanpup3=pd.DataFrame()
dfs = {}

for animal_ID in IDS:
    
    startpoint = time.time()
    teller=teller+1
    print(teller)

    for round1 in range(0,3):        

        if(round1==0):
            dir1='E:\\Dropbox (MIT)\\Baby macaques\\Eyetracking\\firstround\\'
            str1= dir1 + animal_ID + '.h5'
        if(round1==1):
            dir1='E:\\Dropbox (MIT)\\Baby macaques\\Eyetracking\\Second_round\\'
            str1= dir1 + animal_ID + '.h5'
        if(round1==2):
            dir1='E:\\Dropbox (MIT)\\Baby macaques\\Eyetracking\\Mixed_order\\'
            str1= dir1 + animal_ID + '.h5'

        df_fract=pd.read_hdf(str1,'df_fract')
#        df_cum=pd.read_hdf(str1,'df_cum')
        df_cum1=pd.read_hdf(str1,'df_cum1')
        df_pupresult=pd.read_hdf(str1,'df_pupresult')
        df_media=pd.read_hdf(str1,'df_media')
        df_all=pd.read_hdf(str1,'df')
        
        if(round1==0):
            df_fractwatched_1[animal_ID]=[df_fract['submissive'].iloc[0], df_fract['neutral'].iloc[0], df_fract['threat'].iloc[0]]
            df_pupresult_1_ag[animal_ID]=df_pupresult['ag']    
            df_pupresult_1_ne[animal_ID]=df_pupresult['ne']  
            df_pupresult_1_su[animal_ID]=df_pupresult['su']    
            df_cumsum1[animal_ID]=df_cum1['sum_num']
            df_meanpup1[animal_ID]=df_fract['meanpupil']

        if(round1==1):
            df_fractwatched_2[animal_ID]=[df_fract['submissive'].iloc[0], df_fract['neutral'].iloc[0], df_fract['threat'].iloc[0]]
            df_pupresult_2_ag[animal_ID]=df_pupresult['ag']    
            df_pupresult_2_ne[animal_ID]=df_pupresult['ne']  
            df_pupresult_2_su[animal_ID]=df_pupresult['su']                
            df_cumsum2[animal_ID]=df_cum1['sum_num']
            df_meanpup2[animal_ID]=df_fract['meanpupil']            

        if(round1==2):
            df_fractwatched_3[animal_ID]=[df_fract['submissive'].iloc[0], df_fract['neutral'].iloc[0], df_fract['threat'].iloc[0]]
            df_pupresult_3_ag[animal_ID]=df_pupresult['ag']    
            df_pupresult_3_ne[animal_ID]=df_pupresult['ne']  
            df_pupresult_3_su[animal_ID]=df_pupresult['su']                
            df_cumsum3[animal_ID]=df_cum1['sum_num']
            df_meanpup3[animal_ID]=df_fract['meanpupil'] 

            
        
            
#df_fractwatched = pd.concat((df_fractwatched_1, df_fractwatched_2))
#df_fractwatched = df_fractwatched_1
df_concat0 = pd.concat((df_fractwatched_1, df_fractwatched_2), axis=0)
#df_concat0 = pd.concat((df_fractwatched_1, df_fractwatched_2))
d=df_concat0.groupby(level=0).mean()
#df_concat0.mean()

#d[['C1','C2','C3','C4','C5','C6']].mean(axis=1).plot(ax=axes[0], ylim=[0, 1], kind='bar')
#d[['M1','M2','M3','M4','M5']].mean(axis=1).plot(ax=axes[1], ylim=[0, 1], kind='bar')

fig, axes = plt.subplots(nrows=2, ncols=3)
ax1=df_fractwatched_1[['C1','C2','C3','C4','C5','C6']].mean(axis=1).plot(ax=axes[0,0], ylim=[0, 1], kind='bar',title='WT round 1')
ax2=df_fractwatched_1[['M1','M2','M3','M4','M5']].mean(axis=1).plot(ax=axes[1,0], ylim=[0, 1], kind='bar',title='Shank3 round 1')
ax1.set_xticklabels(['sub','neu','thr'],rotation=0)
ax2.set_xticklabels(['sub','neu','thr'],rotation=0)
ax1.set_ylim(0.25,0.65)
ax2.set_ylim(0.45,0.65)

ax3=df_fractwatched_2[['C1','C2','C3','C4','C5','C6']].mean(axis=1).plot(ax=axes[0,1], ylim=[0, 1], kind='bar',title='WT round 2')
ax4=df_fractwatched_2[['M1','M2','M3','M4','M5']].mean(axis=1).plot(ax=axes[1,1], ylim=[0, 1], kind='bar',title='Shank3 round 2')
ax3.set_xticklabels(['sub','neu','thr'],rotation=0)
ax4.set_xticklabels(['sub','neu','thr'],rotation=0)
ax3.set_ylim(0.25,0.65)
ax4.set_ylim(0.45,0.65)
#plt.tight_layout()

ax5=df_fractwatched_3[['C1','C2','C3','C4','C5','C6']].mean(axis=1).plot(ax=axes[0,2], ylim=[0, 1], kind='bar',title='WT round 3')
ax6=df_fractwatched_3[['M1','M2','M3','M4','M5']].mean(axis=1).plot(ax=axes[1,2], ylim=[0, 1], kind='bar',title='Shank3 round 3')
ax5.set_xticklabels(['sub','neu','thr'],rotation=0)
ax6.set_xticklabels(['sub','neu','thr'],rotation=0)
ax5.set_ylim(0.25,0.65)
ax6.set_ylim(0.45,0.65)

plt.suptitle("Fraction watched",fontsize=18)
plt.tight_layout(rect=[0, 0.1, 1, 0.95])



fig, axes = plt.subplots(nrows=1, ncols=2)
m1=d[['C1','C2','C3','C4','C5','C6']].mean(axis=1)
sem1=stats.sem(d[['C1','C2','C3','C4','C5','C6']].mean(axis=1))
ax1=m1.plot(ax=axes[0], ylim=[0, 1], kind='bar',title='WT',yerr=sem1)
m2=d[['M1','M2','M3','M4','M5']].mean(axis=1)
sem2=stats.sem(d[['M1','M2','M3','M4','M5']].mean(axis=1))
ax2=m2.plot(ax=axes[1], ylim=[0, 1], kind='bar',title='Shank3',yerr=sem2)
ax1.set_xticklabels(['sub','neu','thr'],rotation=0)
ax2.set_xticklabels(['sub','neu','thr'],rotation=0)
ax1.set_ylim(0.3,0.6)
ax2.set_ylim(0.3,0.6)
plt.suptitle("Fraction watched, mean",fontsize=18)
plt.tight_layout(rect=[0, 0.1, 1, 0.95])


fig, axes = plt.subplots(nrows=1, ncols=2)
m1=d[['C1','C2','C3','C4','C5','C6']].mean(axis=1)
sem1=stats.sem(d[['C1','C2','C3','C4','C5','C6']].mean(axis=1))
ax1=m1.plot(ax=axes[0], kind='bar',title='WT',yerr=sem1)
m2=d[['M1','M2','M3','M4','M5']].mean(axis=1)
sem2=stats.sem(d[['M1','M2','M3','M4','M5']].mean(axis=1))
ax2=m2.plot(ax=axes[1], kind='bar',title='Shank3',yerr=sem2)
ax1.set_xticklabels(['sub','neu','thr'],rotation=0)
ax2.set_xticklabels(['sub','neu','thr'],rotation=0)
ax1.set_ylim(0.3,0.6)
ax2.set_ylim(0.5,0.6)
plt.suptitle("Fraction watched, mean",fontsize=18)
plt.tight_layout(rect=[0, 0.1, 1, 0.95])






df_pup_ag = pd.concat((df_pupresult_1_ag, df_pupresult_2_ag, df_pupresult_3_ag), axis=0)
d_ag_m=df_pup_ag.groupby(level=0).mean()

df_pup_ne = pd.concat((df_pupresult_1_ne, df_pupresult_2_ne, df_pupresult_3_ne), axis=0)
d_ne_m=df_pup_ne.groupby(level=0).mean()

df_pup_su = pd.concat((df_pupresult_1_su, df_pupresult_2_su, df_pupresult_3_su), axis=0)
d_su_m=df_pup_su.groupby(level=0).mean()

df_concat_pup = pd.concat((d_ag_m, d_ne_m, d_su_m))
foo = df_concat_pup.groupby(level=0).mean()
foo.head()

dfs[0]=d_ag_m
dfs[1]=d_ne_m
dfs[2]=d_su_m
panel = pd.Panel(dfs)
print('Mean of stacked DFs:\n{df}'.format(df=panel.mean(axis=0)))
q=panel.mean(axis=0)

fig, axes = plt.subplots(nrows=1, ncols=2)
ax1_cs=q[['C1','C2','C3','C4','C5','C6']].mean(axis=1).plot(ax=axes[0], kind='line')
ax2_cs=q[['M1','M2','M3','M4','M5']].mean(axis=1).plot(ax=axes[0], kind='line')
axes[0].legend(['WT','Shank3'])
axes[0].title("Pupil response mean",fontsize=18)
plt.tight_layout(rect=[0, 0.1, 1, 0.95])

#for timestep in range(0,115):
#    ttest_ind()
#    
#cat1 = q[q['Category']=='cat1']
#cat2 = q[q['Category']=='cat2']
#
#ttest_ind(cat1['values'], cat2['values'])
#>>> (1.4927289925706944, 0.16970867501294376)






fig, axes = plt.subplots(nrows=1, ncols=2)
ax1ag=d_ag_m[['C1','C2','C3','C4','C5','C6']].mean(axis=1).plot(ax=axes[0], xlim=[0, 50], ylim=[-1, 1], kind='line',title='WT')
ax1ne=d_ne_m[['C1','C2','C3','C4','C5','C6']].mean(axis=1).plot(ax=axes[0], xlim=[0, 50], ylim=[-1, 1], kind='line',title='WT')
ax1su=d_su_m[['C1','C2','C3','C4','C5','C6']].mean(axis=1).plot(ax=axes[0], xlim=[0, 50], ylim=[-1, 1], kind='line',title='WT')

ax2ag=d_ag_m[['M1','M2','M3','M4','M5']].mean(axis=1).plot(ax=axes[1], xlim=[0, 50], ylim=[-1, 1], kind='line',title='Shank3')
ax2ne=d_ne_m[['M1','M2','M3','M4','M5']].mean(axis=1).plot(ax=axes[1], xlim=[0, 50], ylim=[-1, 1], kind='line',title='Shank3')
ax2su=d_su_m[['M1','M2','M3','M4','M5']].mean(axis=1).plot(ax=axes[1], xlim=[0, 50], ylim=[-1, 1], kind='line',title='Shank3')
#ax1.set_xticklabels(['sub','neu','thr'],rotation=0)
#ax2.set_xticklabels(['sub','neu','thr'],rotation=0)
#ax1.set_ylim(0.3,0.4)
#ax2.set_ylim(0.5,0.6)
axes[0].legend(['ag','ne','su'])
axes[1].legend(['ag','ne','su'])
#ax1ne.legend(['ne'])

plt.suptitle("Pupil response",fontsize=18)
plt.tight_layout(rect=[0, 0.3, 1, 0.95])






fig, axes = plt.subplots(nrows=1, ncols=3)
ax1ag=d_ag_m[['C1','C2','C3','C4','C5','C6']].mean(axis=1).plot(ax=axes[0], xlim=[0, 50], ylim=[-1, 1], kind='line',title='AG')
ax1ne=d_ne_m[['C1','C2','C3','C4','C5','C6']].mean(axis=1).plot(ax=axes[1], xlim=[0, 50], ylim=[-1, 1], kind='line',title='NE')
ax1su=d_su_m[['C1','C2','C3','C4','C5','C6']].mean(axis=1).plot(ax=axes[2], xlim=[0, 50], ylim=[-1, 1], kind='line',title='SU')

ax2ag=d_ag_m[['M1','M2','M3','M4','M5']].mean(axis=1).plot(ax=axes[0], xlim=[0, 50], ylim=[-1, 1], kind='line',title='AG')
ax2ne=d_ne_m[['M1','M2','M3','M4','M5']].mean(axis=1).plot(ax=axes[1], xlim=[0, 50], ylim=[-1, 1], kind='line',title='NE')
ax2su=d_su_m[['M1','M2','M3','M4','M5']].mean(axis=1).plot(ax=axes[2], xlim=[0, 50], ylim=[-1, 1], kind='line',title='SU')
#ax1.set_xticklabels(['sub','neu','thr'],rotation=0)
#ax2.set_xticklabels(['sub','neu','thr'],rotation=0)
#ax1.set_ylim(0.3,0.4)
#ax2.set_ylim(0.5,0.6)
axes[0].legend(['WT','Shank3'])
axes[1].legend(['WT','Shank3'])
axes[2].legend(['WT','Shank3'])
#ax1ne.legend(['ne'])

plt.suptitle("Pupil response by content",fontsize=18)
plt.tight_layout(rect=[0, 0.3, 1, 0.95])





df_cs = pd.concat((df_cumsum1, df_cumsum2), axis=0)
df_cs_m=df_cs.groupby(level=0).mean()

fig, axes = plt.subplots(nrows=1, ncols=2)
ax1_cs=df_cs_m[['C1','C2','C3','C4','C5','C6']].mean(axis=1).plot(ax=axes[0], ylim=[3, 4], kind='line')
ax2_cs=df_cs_m[['M1','M2','M3','M4','M5']].mean(axis=1).plot(ax=axes[0], ylim=[3, 4], kind='line')
#ax1.set_xticklabels(['sub','neu','thr'],rotation=0)
#ax2.set_xticklabels(['sub','neu','thr'],rotation=0)
#ax1.set_ylim(0.3,0.4)
#ax2.set_ylim(0.5,0.6)
axes[0].legend(['WT','Shank3'])
#axes[1,0].legend(['ag','ne','su'])
#ax1ne.legend(['ne'])
plt.suptitle("Pupil response mean",fontsize=18)
plt.tight_layout(rect=[0, 0.1, 1, 0.95])


df_mp = pd.concat((df_meanpup1, df_meanpup2), axis=0)
df_mp_m=df_mp.groupby(level=0).mean()




fig, axes = plt.subplots(nrows=1,ncols=2)
m1=df_mp_m[['C1','C2','C3','C4','C5','C6']].mean(axis=1)
sem1=stats.sem(df_mp_m[['C1','C2','C3','C4','C5','C6']],axis=1)
m2=df_mp_m[['M1','M2','M3','M4','M5']].mean(axis=1)
sem2=stats.sem(df_mp_m[['M1','M2','M3','M4','M5']],axis=1)

m3 = pd.DataFrame(columns=[''], index=[0,1])
m3.loc[0] = m1[0]
m3.loc[1] = m2[0]
sem3 = pd.DataFrame(columns=[''], index=[0,1])
sem3.loc[0] = sem1[0]
sem3.loc[1] = sem2[0]
#m3 = pd.DataFrame([[m1, m2]])
#sem3=pd.DataFrame([[sem1, sem2]])

ax2=m3.plot(ax=axes[0], ylim=[3, 4], kind='bar',title='tonic pupil',yerr=sem3,legend=False)
ax2.set_xticklabels(['WT','Shank3'],rotation=0)
#ax2.set_xticklabels(['sub','neu','thr'],rotation=0)
#ax1.set_ylim(0.3,0.4)
#ax2.set_ylim(0.5,0.6)
plt.tight_layout()
a1=df_mp_m.as_matrix(columns={'M1','M2','M3','M4','M5'})
a1=np.reshape(a1, (5, 1))
a2=df_mp_m.as_matrix(columns={'C1','C2','C3','C4','C5','C6'})
a2=np.reshape(a2, (6, 1))
t,p = stats.ttest_ind(a1, a2)  
print("ttest tonic pupil size:            t = %g  p = %g" % (t, p))
#sample1 = np.random.randn(10, 1)
#sample2 = 1 + np.random.randn(15, 1)

#fig, axes = plt.subplots(nrows=2, ncols=2)
#df_media_spc1.plot(ax=axes[0,0], ylim=[0, 1])
#df_media_spc2.plot(ax=axes[1,0], ylim=[0, 1])
#
#df_m1['mean']=df_media_spc1.mean(axis=1)
#df_m2['mean']=df_media_spc2.mean(axis=1)
#
#df_m1.plot(ax=axes[0,1], ylim=[0, 1])
#df_m2.plot(ax=axes[1,1], ylim=[0, 1])
#
#fig, axes = plt.subplots(nrows=2, ncols=2)
#df_media_spc1[['C1','C2','C3','C4','C5','C6']].mean(axis=1).plot(ax=axes[0,0], ylim=[0, 1])
#df_media_spc1[['M1','M2','M3','M4','M5']].mean(axis=1).plot(ax=axes[1,0], ylim=[0, 1])
#
#df_media_spc2[['C1','C2','C3','C4','C5','C6']].mean(axis=1).plot(ax=axes[0,1], ylim=[0, 1])
#df_media_spc2[['M1','M2','M3','M4','M5']].mean(axis=1).plot(ax=axes[1,1], ylim=[0, 1])



#df_media_spc1.to_hdf(str1, 'df_media_spc1', table=True, mode='a',complevel=9)
#df_media_spc2.to_hdf(str1, 'df_media_spc2', table=True, mode='a',complevel=9)
#df_m1.to_hdf(str1, 'df_m1', table=True, mode='a',complevel=9)
#df_m2.to_hdf(str1, 'df_m2', table=True, mode='a',complevel=9)



#df_media_spc1.plot(ax=axes[0,0])
#df_media_spc2.plot(ax=axes[1,0])

#selecting row 1, column 'A'
#df_test['A'].iloc[0]

#a=df_temp1.loc[df_temp1['length'] > ].index.tolist()
#df_temp['block'] = (df_temp.face.shift(1) != df_temp.face).astype(int).cumsum()
#a=df_temp.reset_index().groupby(['face','block'])['index'].apply(np.array)

#We are now almost 7 years after the question was asked, and your code
#
#cells = numpy.array([[0,1,2,3], [2,3,4]])
#executed in numpy 1.12.0, python 3.5, doesn't produce any error and  cells contains:
#
#array([[0, 1, 2, 3], [2, 3, 4]], dtype=object)
#You access your cells elements as cells[0][2] # (=2) .
#
#An alternative to tom10's solution if you want to build your list of numpy arrays on the fly as new elements (i.e. arrays) become available is to use append:
#
#d = []                 # initialize an empty list
#a = np.arange(3)       # array([0, 1, 2])
#d.append(a)            # [array([0, 1, 2])]
#b = np.arange(3,-1,-1) #array([3, 2, 1, 0])
#d.append(b)            #[array([0, 1, 2]), array([3, 2, 1, 0])]