import sys
import numpy as np

def evaluation(diversity,dataset,sensitive_param,model,N):

    DISC_num_Data = []
    accuracy_Data = []
    disc_percentage_Data = []
    time = []
    x_list=[]
    distance_list=[]
#(diversity,dataset,sensitive_param,model,N)
    f = open(dataset + "_" + sensitive_param + "/" + model + "/" + dataset + "_" + sensitive_param + "_" + model + "_" + N + "_" + diversity + ".txt")

    for data in f.readlines():
        D, a, d, t, x, dis = data.split()
        DISC_num_Data.append(float(D))
        accuracy_Data.append(float(a))
        disc_percentage_Data.append(float(d))
        time.append(float(t))
        x_list.append(float(x))
        distance_list.append(float(dis))

    f.close()

    if __name__ == '__main__':
        """
        print "Total evaluated data: "
        print "average: " + str(float(sum(aData) / len(aData)))
        print "standard deviation: " + str(np.std(aData))
        print "max: " + str(max(aData))
        print "min: " + str(min(aData))
        print ""
        """
        """
        average = str(float(sum(DISC_num_Data) / len(DISC_num_Data)))
        std = str(np.std(DISC_num_Data))
        print "DISC_num: " + str(average-1.96*std) + " " + average + " " + str(average+1.96*std)
        print ""
        
        average = str(float(sum(accuracy_Data) / len(accuracy_Data)))
        std = str(np.std(accuracy_Data))
        print "accuracy_Data: " + str(average-1.96*std) + " " + average + " " + str(average+1.96*std)
        print ""
        
        average = str(float(sum(disc_percentage_Data) / len(disc_percentage_Data)))
        std = str(np.std(disc_percentage_Data))
        print "disc_percentage_Data: " + str(average-1.96*std) + " " + average + " " + str(average+1.96*std)
        print ""
        
        average = str(float(sum(time) / len(time)))
        std = str(np.std(time))
        print "time: " + str(average-1.96*std) + " " + average + " " + str(average+1.96*std)
        print ""
        """
    
    average = float(sum(DISC_num_Data) / len(DISC_num_Data))
    std = np.std(DISC_num_Data)
    a = str(average-1.96*std) + " " + str(average) + " " + str(average+1.96*std)
    
    average = float(sum(accuracy_Data) / len(accuracy_Data))
    std = np.std(accuracy_Data)
    b = "before_accuracy: " + str(average-1.96*std) + " " + str(average) + " " + str(average+1.96*std)
     
    average = float(sum(disc_percentage_Data) / len(disc_percentage_Data))
    std = np.std(disc_percentage_Data)
    c = "before_disc_percentage: " + str(average-1.96*std) + " " + str(average) + " " + str(average+1.96*std)
 
    average = float(sum(time) / len(time))
    std = np.std(time)
    d = "after_accuracy: " + str(average-1.96*std) + " " + str(average) + " " + str(average+1.96*std)
    
    average = float(sum(x_list) / len(x_list))
    std = np.std(x_list)
    e = "after_disc_percentage: " + str(average-1.96*std) + " " + str(average) + " " + str(average+1.96*std)
    
    average = float(sum(distance_list) / len(distance_list))
    std = np.std(distance_list)
    f = "distance: " + str(average-1.96*std) + " " + str(average) + " " + str(average+1.96*std)
    
    
    with open(dataset + "_" + sensitive_param + "/" + model + "/" + dataset + "_" + sensitive_param + "_" + model + "_" + N + "_" + diversity + "_evaluation.txt", "w") as myfile:
        myfile.write(a + "\n" + b + "\n" + c + "\n" + d + "\n" + e + "\n" + f + "\n")
                        
                        
if __name__ == '__main__':
    evaluation(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
