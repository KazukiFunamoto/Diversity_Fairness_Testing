import subprocess

subprocess.run("python exp.py --method aft --dataset_name Adult --protected_attr age --model_name SVM --repeat 30 --runtime 3600")
subprocess.run("python exp.py --method aft --dataset_name Adult --protected_attr age --model_name MLP --repeat 30 --runtime 3600")
subprocess.run("python exp.py --method aft --dataset_name Adult --protected_attr age --model_name RanForest --repeat 30 --runtime 3600")

subprocess.run("python exp.py --method aft --dataset_name Adult --protected_attr race --model_name SVM --repeat 30 --runtime 3600")
subprocess.run("python exp.py --method aft --dataset_name Adult --protected_attr race --model_name MLP --repeat 30 --runtime 3600")
subprocess.run("python exp.py --method aft --dataset_name Adult --protected_attr race --model_name RanForest --repeat 30 --runtime 3600")

subprocess.run("python exp.py --method aft --dataset_name Adult --protected_attr sex --model_name SVM --repeat 30 --runtime 3600")
subprocess.run("python exp.py --method aft --dataset_name Adult --protected_attr sex --model_name MLP --repeat 30 --runtime 3600")
subprocess.run("python exp.py --method aft --dataset_name Adult --protected_attr sex --model_name RanForest --repeat 30 --runtime 3600")

subprocess.run("python exp.py --method aft --dataset_name Bank --protected_attr age --model_name SVM --repeat 30 --runtime 3600")
subprocess.run("python exp.py --method aft --dataset_name Bank --protected_attr age --model_name MLP --repeat 30 --runtime 3600")
subprocess.run("python exp.py --method aft --dataset_name Bank --protected_attr age --model_name RanForest --repeat 30 --runtime 3600")

subprocess.run("python exp.py --method aft --dataset_name Credit --protected_attr age --model_name SVM --repeat 30 --runtime 3600")
subprocess.run("python exp.py --method aft --dataset_name Credit --protected_attr age --model_name MLP --repeat 30 --runtime 3600")
subprocess.run("python exp.py --method aft --dataset_name Credit --protected_attr age --model_name RanForest --repeat 30 --runtime 3600")

subprocess.run("python exp.py --method aft --dataset_name Credit --protected_attr sex --model_name SVM --repeat 30 --runtime 3600")
subprocess.run("python exp.py --method aft --dataset_name Credit --protected_attr sex --model_name MLP --repeat 30 --runtime 3600")
subprocess.run("python exp.py --method aft --dataset_name Credit --protected_attr sex --model_name RanForest --repeat 30 --runtime 3600")

