#line count
from os import listdir, chdir
from os.path import isfile, join
mypath='E:\\Dropbox (MIT)\\Zeyun Wu\\labels_txt'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
chdir(mypath)
cumlines=0
ctr=0
for f in onlyfiles:
	num_lines = sum(1 for line in open(f))
	cumlines=cumlines+num_lines
	ctr=ctr+1
	#f.close()
print(cumlines/ctr)
print(cumlines)