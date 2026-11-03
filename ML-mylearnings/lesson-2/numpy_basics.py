import numpy as np


# Water usage data for 5 data centers
water_usage = np.array([
    [850, 900, 870, 920],
    [1200, 1250, 1190, 1300],
    [950, 980, 920, 970],
    [1800, 1750, 1850, 1900],
    [1350, 1400, 1320, 1380]
])


# Display the dataset
print("Water Usage Data:")
print(water_usage)


# Display the shape
print("\nDataset Shape:")
print(water_usage.shape)


# Calculate the overall average
overall_average = np.mean(water_usage)

print("\nOverall Average:")
print(overall_average)


# Calculate the average for each data center
center_averages = np.mean(water_usage, axis=1)

print("\nAverage Usage Per Center:")
print(center_averages)


# Find the highest usage
highest_usage = np.max(water_usage)

print("\nHighest Water Usage:")
print(highest_usage)


# Find the lowest usage
lowest_usage = np.min(water_usage)

print("\nLowest Water Usage:")
print(lowest_usage)


# Add 100 to every measurement
increased_usage = water_usage + 100

print("\nWater Usage After Adding 100:")
print(increased_usage)