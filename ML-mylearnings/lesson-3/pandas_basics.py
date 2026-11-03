import pandas as pd


data = {
    "name": [
        "Center A",
        "Center B",
        "Center C",
        "Center D",
        "Center E"
    ],
    "company": [
        "AWS",
        "Microsoft",
        "Google",
        "Meta",
        "Equinix"
    ],
    "country": [
        "USA",
        "USA",
        "USA",
        "USA",
        "Singapore"
    ],
    "water_usage": [
        850,
        1200,
        950,
        1800,
        1350
    ]
}


df = pd.DataFrame(data)


print("Data:")
print(df)


print("\nDataset Shape:")
print(df.shape)


print("\nAverage Water Usage:")
print(df["water_usage"].mean())


print("\nWater Usage Statistics:")
print(df["water_usage"].describe())


# Challenge 1
usa_centers = df[df["country"] == "USA"]

print("\nUSA Data Centers:")
print(usa_centers)


# Challenge 2
high_usage = df[df["water_usage"] > 1300]

print("\nUsage Above 1300:")
print(high_usage)


# Challenge 3
highest_center = df.sort_values(
    "water_usage",
    ascending=False
).iloc[0]

print("\nHighest Water Usage Center:")
print(highest_center)


# Challenge 4
def classify_usage(usage):
    if usage > 1000:
        return "High"
    else:
        return "Normal"


df["usage_category"] = df["water_usage"].apply(classify_usage)

print("\nData With Usage Category:")
print(df)


# Challenge 5
high_count = (df["usage_category"] == "High").sum()

print("\nNumber of High Usage Centers:")
print(high_count)