# generateCTFiles.py

# Once the txt files from each datasets are created, execute the shell command
# python generateCTFiles.py <dataset> <strength>
import sys
import subprocess

def generateCT(dataset, t):
	tstr = str(t)
	command = '-Ddoi='+tstr+' -Doutput=csv -jar ACTS3.2/acts_3.2.jar CTFiles/'+dataset+'TestModel.txt CTFiles/'+dataset+'/'+dataset+'TS'+tstr+'w.csv'
	subprocess.call('java ' + command, shell=True)

if __name__ == "__main__":
	generateCT(sys.argv[1], sys.argv[2])