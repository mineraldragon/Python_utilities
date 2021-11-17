from scipy.io import wavfile
import os
path = 'C:\\Users\\Rogier\\Documents\\Python Scripts\\acdc-master\\acdc-master\\2018_07_09-01_R1\\trill2'
os.chdir(path)
files = os.listdir(path)
i = 1

for file in files:
	print(file)
	fs, data = wavfile.read(file)
	wavfile.write('a'+file,48000,data)
