import logging
import random
import time
import copy
import itertools
import csv
import os


class Themis:
    def __init__(self, black_box_model, protected_list, show_logging=False):
        self.black_box_model = black_box_model
        self.data_range = self.black_box_model.data_range
        self.protected_list = [self.black_box_model.feature_list[i] for i in protected_list]
        self.protected_list_no = protected_list
        self.no_prot = len(protected_list)
        self.protected_value_comb = self.generate_protected_value_combination()
        self.disc_data = list()
        self.test_data = list()

        self.no_test = 0
        self.no_disc = 0
        self.real_time_consumed = 0
        self.cpu_time_consumed = 0
        if show_logging:
            logging.basicConfig(format="", level=logging.INFO)
        else:
            logging.basicConfig(level=logging.CRITICAL + 1)

    def generate_protected_value_combination(self):
        res = list()
        for index_protected in self.protected_list_no:
            MinMax = self.data_range[index_protected]
            res.append(list(range(MinMax[0], MinMax[1] + 1)))
        return list(itertools.product(*res))

    def check_disc(self, test):
        y = int(self.black_box_model.predict([test]))
        self.no_test += 1
        self.test_data.append(test)

        test2 = copy.deepcopy(test)
        comb_to_be_removed = tuple(test[i] for i in self.protected_list_no)
        comb_removed_same = [item for item in self.protected_value_comb if item != comb_to_be_removed]
        random.shuffle(comb_removed_same)
        for combination in comb_removed_same:
            for i in range(self.no_prot):
                test2[self.protected_list_no[i]] = combination[i]

            y2 = int(self.black_box_model.predict([test2]))
            if y != y2:
                #self.disc_data.append(test+[y])
                #self.disc_data.append(test2+[y2])
                self.disc_data.append(test)
                self.no_disc += 1
                break
                

    def test(self, runtime=None, max_test=10000, max_disc=1000, label=("res",0)):
        data_range = self.data_range
        logging.info(f"Starting fairness test -- {label[0]}")

        start_real_time = time.time()
        start_cpu_time = time.process_time()
        loop = 0
        interval = 1000
        
        method, model, dataset, protected_attr, runtime_str = label[0].split("-")
        
        while True:
            if (runtime is not None) and (time.process_time() - start_cpu_time >= runtime):
                break
            if (max_test is not None) and (loop > max_test):
                break
            if (max_disc is not None) and (self.no_disc >= max_disc):
                break
            if dataset == "Adult" and self.no_disc >= 32561:
                break
            if dataset == "Bank" and self.no_disc >= 45211:
                break
            if dataset == "Credit" and self.no_disc >= 1000:
                break
            loop += 1

            test = list()
            for i in range(self.black_box_model.no_attr):
                test.append(random.randint(data_range[i][0], data_range[i][1]))
            self.check_disc(test)

            #if loop % interval == 0:
            #    logging.info(
            #        f"Loop {loop}: #Disc={self.no_disc}, #Test={self.no_test}, Prec={self.no_disc / self.no_test}")

        self.real_time_consumed = time.time() - start_real_time
        self.cpu_time_consumed = time.process_time() - start_cpu_time
        
        # select random disc data
        if dataset == "Adult" or dataset == "Bank":
            if len(self.disc_data) >= 1000:
                self.disc_data = random.sample(self.disc_data, 1000)
        elif dataset == "Credit":
            if len(self.disc_data) >= 50:
                self.disc_data = random.sample(self.disc_data, 50)

        # save the results of detected discriminatory instances and generated test cases
        
        if model == "SVM":
            model = "SVM"
        elif model == "MLP":
            model ="MLPC"
        elif model == "RanForest":
            model = "RF"
        dataset_and_attr = f"{dataset}-{protected_attr}" 
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
        #output_dir = f"../../Distance/result/{method}/{dataset_and_attr}/{model}"
        output_dir = os.path.join(project_root, "Distance", "result", method, f"{dataset}-{protected_attr}", model)
        if not os.path.exists(output_dir):
            print(f"{output_dir}path")
            raise FileNotFoundError
        disc_file = os.path.join(output_dir, f"{label[0]}-{label[1]}.csv")
   
        logging.info(f"Saving the detected discriminatory instances to {output_dir}/{label[0]}-{label[1]}.csv")
        with open(disc_file, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(self.disc_data)
        logging.info(f"Finished")
        
        
        # check distance      
        def L1_distance(disc1,disc2):
            distance = 0
            for i in range(len(disc1)):
                #print(distance)
                difference = abs(disc1[i]-disc2[i])
                min_val, max_val = self.black_box_model.data_range[i]
                if dataset=="Adult":
                    distance += float(difference)/(max_val - min_val)
                elif dataset=="Bank":
                    distance += float(difference)/(max_val - min_val)
                elif dataset=="Credit":
                    distance += float(difference)/(max_val - min_val)
            return distance
        
        pairwisedistance = 0
        count=0
        disc_num = len(self.disc_data)
        for i in range(0,disc_num-1):
            for j in range(i+1,disc_num):
                pairwisedistance += L1_distance(self.disc_data[i],self.disc_data[j])
                count += 1
        pairwisedistance = float(pairwisedistance) / count
        
        print(count)
        print(pairwisedistance)
        
        # Save pairwise distance
        pairwise_file = os.path.join(output_dir, "pairwise_distance.txt")
        with open(pairwise_file, 'a') as f:
            f.write(f"{pairwisedistance} ({label[0]}-{label[1]})\n")