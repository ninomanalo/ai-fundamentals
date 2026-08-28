import numpy as np
import pandas as pd


water_usage = np.array([
    850,
    1200,
    950,
    1800,
    1350,
    1100,
    1450,
    900,
    1600,
    1250
])


df = pd.DataFrame({
    "water_usage": water_usage
})


print("Water Usage Data:")
print(df)


# Mean
mean_usage = np.mean(water_usage)

print("\nMean:")
print(mean_usage)


# Median
median_usage = np.median(water_usage)

print("\nMedian:")
print(median_usage)


# Minimum
minimum_usage = np.min(water_usage)

print("\nMinimum:")
print(minimum_usage)


# Maximum
maximum_usage = np.max(water_usage)

print("\nMaximum:")
print(maximum_usage)


# Range
usage_range = maximum_usage - minimum_usage

print("\nRange:")
print(usage_range)


# Variance
variance_usage = np.var(water_usage)

print("\nVariance:")
print(variance_usage)


# Standard Deviation
std_usage = np.std(water_usage)

print("\nStandard Deviation:")
print(std_usage)


# Percentiles
q25 = np.percentile(water_usage, 25)
q50 = np.percentile(water_usage, 50)
q75 = np.percentile(water_usage, 75)

print("\n25th Percentile:")
print(q25)

print("\n50th Percentile:")
print(q50)

print("\n75th Percentile:")
print(q75)


# =====================================
# CHALLENGE 1
# Find unusually high water usage
# =====================================

threshold = mean_usage + std_usage

high_usage = water_usage[water_usage > threshold]

print("\nUnusually High Usage:")
print(high_usage)


# =====================================
# CHALLENGE 2
# Find unusually low water usage
# =====================================

threshold_low = mean_usage - std_usage

low_usage = water_usage[water_usage < threshold_low]

print("\nUnusually Low Usage:")
print(low_usage)


# =====================================
# CHALLENGE 3
# Correlation
# =====================================

temperature = np.array([
    28,
    29,
    30,
    35,
    32,
    31,
    34,
    27,
    36,
    33
])


correlation = np.corrcoef(
    temperature,
    water_usage
)[0, 1]

print("\nTemperature / Water Usage Correlation:")
print(correlation)


