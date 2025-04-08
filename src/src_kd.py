import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Đọc dữ liệu từ file CSV
df = pd.read_csv("data/Cleaned_GPUs_KNN.csv")

# Lấy mẫu ngẫu nhiên 100 dòng
sample_df = df.sample(n=100, random_state=42)

# -------------------------------
# 1. Kiểm định một mẫu (One-sample t-test)
# -------------------------------
core_speeds = sample_df['Core_Speed']
hypothesized_mean = 700  # Giả thuyết trung bình

# Kiểm định
t_stat_1samp, p_value_1samp = stats.ttest_1samp(core_speeds, popmean=hypothesized_mean)

# In kết quả
print("One-sample t-test:")
print(f"  Trung bình mẫu: {core_speeds.mean():.2f}")
print(f"  t-statistic: {t_stat_1samp:.3f}")
print(f"  p-value: {p_value_1samp:.4f}")
print("  =>", "Bác bỏ H0" if p_value_1samp < 0.05 else "Không bác bỏ H0")
print()

# -------------------------------
# 2. Kiểm định hai mẫu (Two-sample t-test)
# -------------------------------
# Lấy 30 mẫu từ mỗi hãng
nvidia_speed = sample_df[sample_df['Manufacturer'] == 'Nvidia']['Core_Speed'][:30]
amd_speed = sample_df[sample_df['Manufacturer'] == 'AMD']['Core_Speed'][:30]

# Kiểm định
t_stat_2samp, p_value_2samp = stats.ttest_ind(nvidia_speed, amd_speed, equal_var=False)

# In kết quả
print("Two-sample t-test:")
print(f"  Trung bình Nvidia: {nvidia_speed.mean():.2f}")
print(f"  Trung bình AMD: {amd_speed.mean():.2f}")
print(f"  t-statistic: {t_stat_2samp:.3f}")
print(f"  p-value: {p_value_2samp:.4f}")
print("  =>", "Bác bỏ H0" if p_value_2samp < 0.05 else "Không bác bỏ H0")

# -------------------------------
# Biểu đồ trực quan
# -------------------------------
sns.set(style="whitegrid")
plt.figure(figsize=(12, 5))

# One-sample: Histogram + KDE
plt.subplot(1, 2, 1)
sns.histplot(core_speeds, kde=True, color='skyblue', bins=15)
plt.axvline(hypothesized_mean, color='red', linestyle='--', label=f'Giá trị giả thuyết = {hypothesized_mean}')
plt.axvline(core_speeds.mean(), color='green', linestyle='-', label=f'Trung bình mẫu = {core_speeds.mean():.2f}')
plt.title('One-sample t-test: Core Speed vs. 700 MHz')
plt.xlabel('Core Speed (MHz)')
plt.legend()

# Two-sample: Boxplot
plt.subplot(1, 2, 2)
two_sample_df = pd.DataFrame({
    'Core_Speed': pd.concat([nvidia_speed, amd_speed], ignore_index=True),
    'Manufacturer': ['Nvidia'] * len(nvidia_speed) + ['AMD'] * len(amd_speed)
})
sns.boxplot(x='Manufacturer', y='Core_Speed', data=two_sample_df, palette='Set2')
plt.title('Two-sample t-test: Nvidia vs. AMD (Core Speed)')
plt.ylabel('Core Speed (MHz)')

plt.tight_layout()
plt.savefig("Chart/Kiem_Dinh/1_and_2_samples_test.png")
plt.show()
