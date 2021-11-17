import csv
csv_file = 'C:\\Users\\Rogier\\Documents\\Python Scripts\\acdc-master\\acdc-master\\2018_07_09-01_R2\\results\\20180814_133248_june29sel\\june29sel.csv'
txt_file = 'C:\\Users\\Rogier\\Documents\\Python Scripts\\acdc-master\\acdc-master\\2018_07_09-01_R2\\results\\20180814_133248_june29sel\\labels_types.txt'

file = open(txt_file,"a") 

with open(csv_file, newline='') as csvfile1:
	readCSV = csv.reader(csvfile1)
	next(readCSV)
	for row in readCSV:
		if any(row):
			p=[row[1], row[2], row[0]]
			print(p)
#			if row[0]:
			file.write(row[1] + "\t" + row[2] + "\t" + row[0] + "\n")			
file.close()
