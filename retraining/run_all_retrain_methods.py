import subprocess

execution_times = 30

for i in range(execution_times):

    subprocess.call("python retrain_methods.py all BANK_age SVM 10000")
    subprocess.call("python retrain_methods.py all BANK_age MLPC 10000")
    subprocess.call("python retrain_methods.py all BANK_age RF 1000") 
    
    subprocess.call("python retrain_methods.py all CENSUS_age SVM 1000")
    subprocess.call("python retrain_methods.py all CENSUS_age MLPC 1000")
    subprocess.call("python retrain_methods.py all CENSUS_age RF 1000") 
    
    subprocess.call("python retrain_methods.py all CENSUS_race SVM 10000")
    subprocess.call("python retrain_methods.py all CENSUS_race MLPC 10000") 
    subprocess.call("python retrain_methods.py all CENSUS_race RF 1000") 
    
    subprocess.call("python retrain_methods.py all CENSUS_sex SVM 100000")
    subprocess.call("python retrain_methods.py all CENSUS_sex MLPC 10000")
    subprocess.call("python retrain_methods.py all CENSUS_sex RF 1000 ") 
    
    subprocess.call("python retrain_methods.py all GERMAN_age SVM 1000")
    subprocess.call("python retrain_methods.py all GERMAN_age MLPC 1000")
    subprocess.call("python retrain_methods.py all GERMAN_age RF 1000")
    
    subprocess.call("python retrain_methods.py all GERMAN_sex SVM 10000") 
    subprocess.call("python retrain_methods.py all GERMAN_sex MLPC 1000")
    subprocess.call("python retrain_methods.py all GERMAN_sex RF 1000")


