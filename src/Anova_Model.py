import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
# import statsmodels.api as sm
# import statsmodels.formula.api as smf

# Load the dataset
df = pd.read_csv("data/Cleaned_GPUs_KNN.csv")
AMD = df[df["Manufacturer"] == "AMD"]["Memory_Bandwidth"]
Intel = df[df["Manufacturer"] == "Intel"]["Memory_Bandwidth"]
ATI = df[df["Manufacturer"] == "ATI"]["Memory_Bandwidth"]
Nvidia = df[df["Manufacturer"] == "Nvidia"]["Memory_Bandwidth"]

# Plot Q-Q plots
fig, axes = plt.subplots(2, 2, tight_layout=True)
axes = axes.flatten()  # Flatten in case of 2D array
manufacturers = df["Manufacturer"].unique() # Get unique manufacturers
for i, manufacturer in enumerate(manufacturers):
    data = df[df["Manufacturer"] == manufacturer]["Memory_Bandwidth"]
    stats.probplot(data, dist="norm", plot=axes[i])
    # nbins = len(data)
    # axes[i].hist(data.values, bins=nbins , alpha=0.3, linewidth=1, edgecolor="white")
    axes[i].set_title(f"{manufacturer}")
    axes[i].grid(True, alpha=0.3)
plt.savefig('Chart/Mo_Hinh_Anova/Memory_Bandwidth Distribution according to Manufacturer')
plt.show()

# shapiro-wilk tests
for i, manufacturer in enumerate(manufacturers):
    data = df[df["Manufacturer"] == manufacturer]["Memory_Bandwidth"]
    shapiro_test = stats.shapiro(data.values)
    print(f"{manufacturer}: {shapiro_test}")

# group the manufacturers 
groups = [group["Memory_Bandwidth"].to_numpy() for name, group in df.groupby(["Manufacturer"])]

# same variance test
stat, p = stats.levene(*groups)
print(f"Levene's Test Statistic: {stat}")
print(f"P-value: {p}")

# ANOVA
f_stat, p_value = stats.f_oneway(*groups)
print(f"F-statistic: {f_stat}")
print(f"P-value: {p_value}")

# Perform the Kruskal-Wallis H-test
h_statistic, p_value = stats.kruskal(AMD, Intel, ATI, Nvidia)
k = len(df["Manufacturer"].unique()) # Calculate the degrees of freedom
df = k - 1
print("H-statistic:", h_statistic)
print("p-value:", p_value)
print("Degrees of freedom:", df)

# tukey's hsd test 
res = stats.tukey_hsd(AMD, Intel, ATI, Nvidia)
print(res)
plt.title("Tukey HSD: Memory Bandwidth by Manufacturer")
plt.xlabel("Mean Difference")
plt.grid(True, alpha=0.3)