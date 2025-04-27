subprocess.call("python diversity_retrain.py BANK_age SVM 10000")
subprocess.call("python diversity_retrain.py BANK_age MLPC 10000")
subprocess.call("python diversity_retrain.py BANK_age RF 1000") 
changed by


subprocess.call("python retrain_methods.py all BANK_age SVM 10000")
subprocess.call("python retrain_methods.py all BANK_age MLPC 10000")
subprocess.call("python retrain_methods.py all BANK_age RF 1000")
changed by
python retrain_methods.py all BANK_age SVM