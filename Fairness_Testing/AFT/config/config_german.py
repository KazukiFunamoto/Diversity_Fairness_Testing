# ORIGINAL FILE
params = 20

sensitive_param_sex = 9     # Starts at 1.
sensitive_param_age = 13

input_bounds = []

input_bounds.append([1, 4])
input_bounds.append([4, 72])
input_bounds.append([0, 4])
input_bounds.append([0, 10])
input_bounds.append([2, 184])
input_bounds.append([1, 5])
input_bounds.append([1, 5])
input_bounds.append([1, 4])
input_bounds.append([0, 1])     #sex (Discriminatory parameter)
input_bounds.append([1, 3])
input_bounds.append([1, 4])
input_bounds.append([1, 4])
input_bounds.append([1, 7])     #age
input_bounds.append([1, 3])
input_bounds.append([1, 3])
input_bounds.append([1, 4])
input_bounds.append([1, 4])
input_bounds.append([1, 2])
input_bounds.append([1, 2])
input_bounds.append([1, 2])

threshold = 0

perturbation_unit = 1

retraining_inputs = "Retrain_Example_File.txt"