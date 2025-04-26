# ORIGINAL FILE
params = 13

sensitive_param_age = 1
sensitive_param_race = 8
sensitive_param_sex = 9     # Starts at 1.

integer_param = [1,3,10,11,12] 
categorical_param = [2,4,5,6,7,8,13]

input_bounds = []

input_bounds.append([1, 9])     # age
input_bounds.append([0, 7])
input_bounds.append([0, 39])
input_bounds.append([0, 15])   #(ecucation) #education-num is deleted
input_bounds.append([0, 6])    #(material status)
input_bounds.append([0, 13])
input_bounds.append([0, 5])
input_bounds.append([0, 4])     # race
input_bounds.append([0, 1])     # sex (Discriminatory parameter)
input_bounds.append([0, 99])
input_bounds.append([0, 39])
input_bounds.append([0, 99])
input_bounds.append([0, 39])

threshold = 0

perturbation_unit = 1

retraining_inputs = "Retrain_Example_File.txt"