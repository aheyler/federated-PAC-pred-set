import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_arr = np.load("/home/aheyler/snapshots/latest_femnist_ps_v4_nov1/logits_val_arr.npy")
test_arr = np.load("/home/aheyler/snapshots/latest_femnist_ps_v4_nov1/logits_test_arr.npy")
df = pd.DataFrame.from_dict({"val logits": train_arr, "test logits": test_arr})
print(df)

df.to_csv("/home/aheyler/PAC-pred-set/snapshots/latest_femnist_ps_v4_nov1/debug_logits.csv")

plt.figure()
plt.hist(train_arr)
plt.show()