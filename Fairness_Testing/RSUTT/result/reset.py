import os
import sys
import shutil

num = int(sys.argv[1])

if num == 1:
    algorithm = ["AEQUITAS", "KOSEI", "CGFT", "RSUTT"]
    # classifier = ["DT", "MLPC", "RF"]
    # dataset = ["CENSUS", "GERMAN", "BANK"]


for a in algorithm:
    shutil.rmtree(a)
    os.mkdir(a)

    print "reset"

else:
    print "not reset"
