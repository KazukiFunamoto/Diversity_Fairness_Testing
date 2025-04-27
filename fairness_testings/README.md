subprocess.run("python exp.py --method themis --dataset_name Adult --protected_attr age --model_name SVM --repeat 30 --runtime 3600")
subprocess.run("python exp.py --method themis --dataset_name Adult --protected_attr age --model_name MLP --repeat 30 --runtime 3600")
subprocess.run("python exp.py --method themis --dataset_name Adult --protected_attr age --model_name RanForest --repeat 30 --runtime 3600")

subprocess.run("python exp.py --method aft --dataset_name Adult --protected_attr age --model_name SVM --repeat 30 --runtime 3600")
subprocess.run("python exp.py --method aft --dataset_name Adult --protected_attr age --model_name MLP --repeat 30 --runtime 3600")
subprocess.run("python exp.py --method aft --dataset_name Adult --protected_attr age --model_name RanForest --repeat 30 --runtime 3600")

subprocess.call("python RSUTT.py RSUTT BANK_age SVM 10000 30")
subprocess.call("python RSUTT.py RSUTT BANK_age MLPC 10000 30")
subprocess.call("python RSUTT.py RSUTT BANK_age RF 1000 30") 
changed by
python RSUTT.py RSUTT BANK_age SVM 30
python RSUTT.py RSUTT CENSUS_sex MLPC 30
python RSUTT.py RSUTT GERMAN_sex RF 30
