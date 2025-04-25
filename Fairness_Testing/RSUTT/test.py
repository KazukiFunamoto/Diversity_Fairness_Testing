import subprocess

subprocess.call("python RSUTT.py RSUTT CENSUS_age MLPC 1000 10")

subprocess.call("python RSUTT.py RSUTT BANK_age SVM 10000 10")
subprocess.call("python RSUTT.py RSUTT BANK_age MLPC 10000 10")
subprocess.call("python RSUTT.py RSUTT BANK_age RF 1000 10") 

subprocess.call("python RSUTT.py RSUTT CENSUS_age SVM 1000 10")
#subprocess.call("python RSUTT.py RSUTT CENSUS_age MLPC 1000 10")
subprocess.call("python RSUTT.py RSUTT CENSUS_age RF 1000 10") 

subprocess.call("python RSUTT.py RSUTT CENSUS_race SVM 10000 10")
subprocess.call("python RSUTT.py RSUTT CENSUS_race MLPC 10000 10") 
subprocess.call("python RSUTT.py RSUTT CENSUS_race RF 1000 10") 

subprocess.call("python RSUTT.py RSUTT CENSUS_sex SVM 100000 10")
subprocess.call("python RSUTT.py RSUTT CENSUS_sex MLPC 10000 10")
subprocess.call("python RSUTT.py RSUTT CENSUS_sex RF 1000 10") 

subprocess.call("python RSUTT.py RSUTT GERMAN_age SVM 1000 10")
subprocess.call("python RSUTT.py RSUTT GERMAN_age MLPC 1000 10")
subprocess.call("python RSUTT.py RSUTT GERMAN_age RF 1000 10")

subprocess.call("python RSUTT.py RSUTT GERMAN_sex SVM 10000 10") 
subprocess.call("python RSUTT.py RSUTT GERMAN_sex MLPC 1000 10")
subprocess.call("python RSUTT.py RSUTT GERMAN_sex RF 1000 10")


"""
subprocess.call("python RSUTT.py RSUTT CENSUS_age SVM 3000 30")
subprocess.call("python RSUTT.py RSUTT CENSUS_age MLPC 3000 30")
subprocess.call("python RSUTT.py RSUTT CENSUS_age RF 3000 30")

subprocess.call("python RSUTT.py RSUTT CENSUS_race SVM 3000 30")
subprocess.call("python RSUTT.py RSUTT CENSUS_race MLPC 3000 30")
subprocess.call("python RSUTT.py RSUTT CENSUS_race RF 3000 30")

subprocess.call("python RSUTT.py RSUTT CENSUS_sex SVM 3000 30")
subprocess.call("python RSUTT.py RSUTT CENSUS_sex MLPC 3000 30")
subprocess.call("python RSUTT.py RSUTT CENSUS_sex RF 3000 30")

subprocess.call("python RSUTT.py RSUTT BANK_age SVM 3000 30")
subprocess.call("python RSUTT.py RSUTT BANK_age MLPC 3000 30")
subprocess.call("python RSUTT.py RSUTT BANK_age RF 3000 30")

subprocess.call("python RSUTT.py RSUTT GERMAN_age SVM 3000 30")
subprocess.call("python RSUTT.py RSUTT GERMAN_age MLPC 3000 30")
subprocess.call("python RSUTT.py RSUTT GERMAN_age RF 3000 30")

subprocess.call("python RSUTT.py RSUTT GERMAN_sex SVM 3000 30")
subprocess.call("python RSUTT.py RSUTT GERMAN_sex MLPC 3000 30")
subprocess.call("python RSUTT.py RSUTT GERMAN_sex RF 3000 30")
"""