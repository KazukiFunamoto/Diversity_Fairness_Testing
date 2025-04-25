import os

root_dir = '.'
output_file = os.path.join(root_dir, 'average_pairwise_distance.txt')  

results = []

for dataset_dir in os.listdir(root_dir):
    dataset_path = os.path.join(root_dir, dataset_dir)
    if not os.path.isdir(dataset_path):
        continue

    for model_dir in os.listdir(dataset_path):
        model_path = os.path.join(dataset_path, model_dir)
        if not os.path.isdir(model_path):
            continue

        pairwise_file = os.path.join(model_path, 'pairwise_distance.txt')
        if not os.path.exists(pairwise_file):
            print("Warning: {} not found. Skipping.".format(pairwise_file))
            continue

        with open(pairwise_file, 'r') as f:
            distances = []
            for line in f:
                try:
                    value = float(line.split()[0])
                    distances.append(value)
                except:
                    continue

        if distances:
            average = sum(distances) / len(distances)
            label = "{:.2f} (rsutt_noncut {} {})".format(average, dataset_dir, model_dir)
            results.append(label)


with open(output_file, 'w') as f:
    for line in results:
        f.write(line + '\n')

print("Average pairwise distances written to {}".format(output_file))
