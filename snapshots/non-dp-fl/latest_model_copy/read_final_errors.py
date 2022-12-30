import numpy as np

train = np.load("/home/aheyler/PAC-pred-set/snapshots/non-dp-fl/latest_model_copy/training_errors copy.npy")
test = np.load("/home/aheyler/PAC-pred-set/snapshots/non-dp-fl/latest_model_copy/test_errors copy.npy")
# val = np.load("/home/aheyler/PAC-pred-set/snapshots/non-dp-fl/latest_model_copy/val_errors copy.npy")

print(f"min train = {np.min(train)}")
print(f"min test = {np.min(test)}")
# print(f"min val = {np.min(val)}")
