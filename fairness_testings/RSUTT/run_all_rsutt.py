import subprocess

dataset_protected_pairs = [
    ("BANK", "age"),
    ("CENSUS", "age"),
    ("CENSUS", "race"),
    ("CENSUS", "sex"),
    ("GERMAN", "age"),
    ("GERMAN", "sex"),
]

models = ["SVM", "MLPC", "RF"]

for dataset, attr in dataset_protected_pairs:
    for model in models:
        cmd = "python RSUTT.py RSUTT {}_{} {} 30".format(dataset, attr, model)
        subprocess.call(cmd, shell=True)
