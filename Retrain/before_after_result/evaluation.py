import sys
import numpy as np

def evaluation(algorithm, dataset, classifier, N):
    
    disc_num_Data = []
    before_accuracy_Data = []
    before_fairness_Data = []
    after_accuracy_Data = []
    after_fairness_Data = []
    
    f = open(algorithm + "/" + dataset + "_" + classifier + "_" + N + ".txt")

    for data in f.readlines():
        dn, ba, bf, aa, af = data.split()
        disc_num_Data.append(float(dn))
        before_accuracy_Data.append(float(ba))
        before_fairness_Data.append(float(bf))
        after_accuracy_Data.append(float(aa))
        after_fairness_Data.append(float(af))

    f.close()

    if __name__ == '__main__':
        """
        print "Total evaluated data: "
        print "average: " + str(float(sum(aData) / len(aData)))
        print "standard deviation: " + str(np.std(aData))
        print "max: " + str(max(aData))
        print "min: " + str(min(aData))
        print ""

        print "Number of seed data: "
        print "average: " + str(float(sum(sData) / len(sData)))
        print "standard deviation: " + str(np.std(sData))
        print "max: " + str(max(sData))
        print "min: " + str(min(sData))
        print ""

        print "Number of discriminatory data: "
        print "average: " + str(float(sum(dData) / len(dData)))
        print "standard deviation: " + str(np.std(dData))
        print "max: " + str(max(dData))
        print "min: " + str(min(dData))
        print ""

        print "Percentage of discriminatory data: "
        print "average: " + str(float(sum(dDataPercent) / len(dDataPercent)))
        print "standard deviation: " + str(np.std(dDataPercent))
        print "max: " + str(max(dDataPercent))
        print "min: " + str(min(dDataPercent))
        print ""

        print "Number of discriminatory data per second: "
        print "average: " + str(float(sum(dDataPerSecond) / len(dDataPerSecond)))
        print "standard deviation: " + str(np.std(dDataPerSecond))
        print "max: " + str(max(dDataPerSecond))
        print "min: " + str(min(dDataPerSecond))
        print ""

        print "Execution_time: "
        print "average: " + str(float(sum(time) / len(time)))
        print "standard deviation: " + str(np.std(time))
        print "max: " + str(max(time))
        print "min: " + str(min(time))
        print ""
        """

    average = float(sum(disc_num_Data) / len(disc_num_Data))
    std = np.std(disc_num_Data)
    a = "disc_num_Data: " + str(average-1.96*std) + " " + str(average) + " " + str(average+1.96*std)
     
    average = float(sum(before_accuracy_Data) / len(before_accuracy_Data))
    std = np.std(before_accuracy_Data)
    b = "before_accuracy_Data: " + str(average-1.96*std) + " " + str(average) + " " + str(average+1.96*std)
    
    average = float(sum(before_fairness_Data) / len(before_fairness_Data))
    std = np.std(before_fairness_Data)
    c = "before_fairness_Data: " + str(average-1.96*std) + " " + str(average) + " " + str(average+1.96*std)
    
    average = float(sum(after_accuracy_Data) / len(after_accuracy_Data))
    std = np.std(after_accuracy_Data)
    d = "after_accuracy_Data: " + str(average-1.96*std) + " " + str(average) + " " + str(average+1.96*std)
    
    average = float(sum(after_fairness_Data) / len(after_fairness_Data))
    std = np.std(after_fairness_Data)
    e = "after_fairness_Data: " + str(average-1.96*std) + " " + str(average) + " " + str(average+1.96*std)
    
    
    with open(algorithm + "/" + dataset + "_" + classifier + "_" + N + "_evaluation.txt", "w") as myfile:
        myfile.write(a + "\n" + b + "\n" + c + "\n" + d + "\n" + e + "\n")

    


if __name__ == '__main__':
    evaluation(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
