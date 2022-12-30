import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, ranksums

set = np.load("/home/aheyler/PAC-pred-set/figures/set.npy")
membership = np.load("/home/aheyler/PAC-pred-set/figures/membership.npy")
sizes = np.load("/home/aheyler/PAC-pred-set/figures/sizes.npy")
labels = np.load("/home/aheyler/PAC-pred-set/figures/labels.npy")

df = pd.DataFrame.from_dict({"membership": membership, "size": sizes, "label": labels})
incorrect_df = df[df["membership"] == False]
full_correct_df = df[df["membership"] == True]
correct_df = full_correct_df.sample(len(incorrect_df))
print(len(incorrect_df))
print(len(correct_df))
df = pd.concat([correct_df, incorrect_df])

plt.figure()
sns.boxplot(x="membership", y="size", data=df)
plt.xlabel("Correct Model Prediction")
plt.ylabel("Size of Prediction Set")
plt.title("Prediction Set Size for Correct vs. Incorrect Predictions (ε=0.01, δ=0.001)")
plt.savefig("/home/aheyler/PAC-pred-set/figures/size_boxplot.png")

print(np.min(correct_df["size"]), np.quantile(correct_df["size"], 0.25), np.median(correct_df["size"]), 
      np.quantile(correct_df["size"], 0.75), np.max(correct_df["size"]), np.mean(correct_df["size"]))

print(np.min(incorrect_df["size"]), np.quantile(incorrect_df["size"], 0.25), np.median(incorrect_df["size"]), 
      np.quantile(incorrect_df["size"], 0.75), np.max(incorrect_df["size"]), np.mean(incorrect_df["size"]))

print(ttest_ind(incorrect_df["size"], full_correct_df["size"], equal_var=False))
print(ranksums(correct_df["size"], incorrect_df["size"], alternative="less"))
print(ranksums(full_correct_df["size"], incorrect_df["size"], alternative="less"))
print(np.var(full_correct_df["size"]))
print(np.var(incorrect_df["size"]))


plt.figure()
sns.histplot(full_correct_df["size"])
plt.savefig("/home/aheyler/PAC-pred-set/figures/sizes_hist_correct.png")

plt.figure()
sns.histplot(incorrect_df["size"])
plt.savefig("/home/aheyler/PAC-pred-set/figures/sizes_hist_incorrect.png")