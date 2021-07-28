#Generate label partitioning for multi label classification"

import numpy as np

import dill as pickle
import sys

def get_random_label_part(n_labels=68, random_state = 42):
	np.random.seed(random_state)
	lab_part = np.random.choice(list(range(n_labels)), size = (int(n_labels/4),4), replace = False)
	return lab_part

##Add: Generate data-driven label partitioning


if __name__ == "__main__":
	for i in range(10):
		lab_part = get_random_label_part(n_labels=68, random_state=i)
		with open(f"random_label_partitioning_{i:02d}", 'wb') as f:
			pickle.dump(lab_part, f)
	sys.exit()