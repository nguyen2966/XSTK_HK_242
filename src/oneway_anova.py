import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
import seaborn as sns
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Load the dataset
df = pd.read_csv("data/Cleaned_GPUs_KNN.csv")
Nvidia = df[df["Manufacturer"] == "Nvidia"]["Memory_Speed"]
AMD = df[df["Manufacturer"] == "AMD"]["Memory_Speed"]
Intel = df[df["Manufacturer"] == "Intel"]["Memory_Speed"]
ATI = df[df["Manufacturer"] == "ATI"]["Memory_Speed"]

# Group the manufacturers 
# groups = [group["Memory_Speed"].to_numpy() for name, group in df.groupby(["Manufacturer"])]

# Plot Q-Q plots
fig, axes = plt.subplots(2, 2, tight_layout=True)
axes = axes.flatten()  # Flatten in case of 2D array
manufacturers = df["Manufacturer"].unique() # Get unique manufacturers
for i, manufacturer in enumerate(manufacturers):
    data = df[df["Manufacturer"] == manufacturer]["Memory_Speed"]
    stats.probplot(data, dist="norm", plot=axes[i])
    # nbins = len(data) 
    # axes[i].hist(data.values, bins=nbins , alpha=0.5, linewidth=5, edgecolor="white")
    axes[i].set_title(f"{manufacturer}")
    axes[i].grid(True, alpha=0.3)
plt.savefig("Chart/Mo_Hinh_Anova/QQ Plot")
plt.show()

# Box plot
fig, ax = plt.subplots(1, 1)
ax.boxplot([Nvidia, AMD, Intel, ATI], showmeans=True)
ax.set_xticklabels(["Nvidia", "AMD", "Intel", "ATI"]) 
ax.set_ylabel("MMemory_Speed (MHz)") 
# sns.boxplot(data=df, x="Manufacturer", y="Memory_Speed", showmeans=True)
plt.savefig("Chart/Mo_Hinh_Anova/Box plot")
plt.show()

# Shapiro-Wilk tests
print("SHAPIRO-WILK TEST")
print(f"Manufacturer: Test Statistic P-value")
for i, manufacturer in enumerate(manufacturers):
    data = df[df["Manufacturer"] == manufacturer]["Memory_Speed"]
    statistic, p_value = stats.shapiro(data)
    print(f"{manufacturer:<15}: Test Statistic = {statistic} P-value = {p_value}")
print()

# Variance and standard deviation
print("VARIANCE AND STANDARD DEVIATION")
print("VARIANCE")
group_variance = df.groupby(["Manufacturer"])["Memory_Speed"].var()
print(AMD.var())
print(group_variance)
print()
print("STANDARD DEVIATION")
group_deviation = df.groupby(["Manufacturer"])["Memory_Speed"].std()
print(group_deviation)
print()

# Same variance test
print("TEST VARIANCE EQUALITY")
statistic, p_value = stats.levene(Nvidia, AMD, Intel, ATI)
print(f"Levene's Test Statistic: {statistic}")
print(f"P-value: {p_value}")
print()

# ANOVA
print("ANOVA")
# f_stat, p_value = stats.f_oneway(AMD, Intel, ATI, Nvidia)
# print(f"F Statistic: {f_stat}")
# print(f"P-value: {p_value}")
model = ols('Memory_Speed ~ Manufacturer', data = df).fit()           
anova_result = sm.stats.anova_lm(model, typ=2)
print(anova_result)
print()

# Perform the Kruskal-Wallis H-test
print("KRUSKAL-WALLIS")
h_statistic, p_value = stats.kruskal(Nvidia, AMD, Intel, ATI)
degree_of_freedom = len(df["Manufacturer"].unique()) - 1 # Calculate the degrees of freedom
print("H Statistic:", h_statistic)
print("P-value:", p_value)
print("Degrees of freedom:", degree_of_freedom)
print()

# Tukey's hsd test 
print("TUKEY'S HSD")
#print("0: Nvidia; 1: AMD; 2: Intel; 3: ATI")
# tukey_result = stats.tukey_hsd(Nvidia, AMD, Intel, ATI)
# print(tukey_result)
# plt.title("Tukey's HSD: Memory_Speed by Manufacturer")
# plt.xlabel("Least Significant Difference")
# plt.grid(True, alpha=0.3)
tukey_result = pairwise_tukeyhsd(endog=df["Memory_Speed"], groups=df["Manufacturer"], alpha=0.05)   
print(tukey_result.summary())
print()