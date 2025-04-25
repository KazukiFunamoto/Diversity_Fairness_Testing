import subprocess

execution_times = 30

for i in range(execution_times):
    subprocess.call("python diversity_test.py BANK_age SVM 10000")
    subprocess.call("python diversity_test.py BANK_age MLPC 10000")
    subprocess.call("python diversity_test.py BANK_age RF 1000") 
    
    subprocess.call("python diversity_test.py CENSUS_age SVM 1000")
    subprocess.call("python diversity_test.py CENSUS_age MLPC 1000")
    subprocess.call("python diversity_test.py CENSUS_age RF 1000") 
    
    subprocess.call("python diversity_test.py CENSUS_race SVM 10000")
    subprocess.call("python diversity_test.py CENSUS_race MLPC 10000") 
    subprocess.call("python diversity_test.py CENSUS_race RF 1000") 
    
    subprocess.call("python diversity_test.py CENSUS_sex SVM 100000 ")
    subprocess.call("python diversity_test.py CENSUS_sex MLPC 10000 ")
    subprocess.call("python diversity_test.py CENSUS_sex RF 1000 ") 
    
    subprocess.call("python diversity_test.py GERMAN_age SVM 1000")
    subprocess.call("python diversity_test.py GERMAN_age MLPC 1000")
    subprocess.call("python diversity_test.py GERMAN_age RF 1000")
    
    subprocess.call("python diversity_test.py GERMAN_sex SVM 10000") 
    subprocess.call("python diversity_test.py GERMAN_sex MLPC 1000")
    subprocess.call("python diversity_test.py GERMAN_sex RF 100")


