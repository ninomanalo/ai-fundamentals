import pandas as pd


data = {
    "name": [
        "Center A",
        "Center B",
        "Center C",
        "Center C",
        "Center D",
        "Center E"
    ],
    "company": [
        "AWS",
        "Microsoft",
        " Google ",
        " Google ",
        "Meta",
        "Equinix"
    ],
    "country": [
        "USA",
        "USA",
        "USA",
        "USA",
        None,
        "Singapore"
    ],
    "water_usage": [
        850,
        1200,
        950,
        950,
        None,
        1350
    ]
}


df = pd.DataFrame(data)


print("Original Data:")
print(df)


print("\nMissing Values:")
print(df.isnull().sum())


df["country"] = df["country"].fillna("Unknown")


df["water_usage"] = df["water_usage"].fillna(
    df["water_usage"].mean()
)


df = df.drop_duplicates()


df["company"] = df["company"].str.strip()


df = df.reset_index(drop=True)


print("\nCleaned Data:")
print(df)


print("\nFinal Missing Values:")
print(df.isnull().sum())


new_center = pd.DataFrame({
    "name": ["Center F"],
    "company": ["Amazon"],
    "country": ["USA"],
    "water_usage": [2000]
})

df = pd.concat([df, new_center], ignore_index=True)

print("\nAfter Adding Center F:")
print(df)



def classify_usage(usage):
    if usage > 1000:
        return "High"
    else:
        return "Normal"


df["usage_category"] = df["water_usage"].apply(classify_usage)

print("\nUsage Category:")
print(df)



highest_usage = df["water_usage"].max()

print("\nHighest Water Usage:")
print(highest_usage)



average_by_company = df.groupby("company")["water_usage"].mean()

print("\nAverage Water Usage by Company:")
print(average_by_company)