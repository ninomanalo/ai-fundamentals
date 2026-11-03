import pandas as pd
import matplotlib.pyplot as plt


data = {
    "name": [
        "Center A",
        "Center B",
        "Center C",
        "Center D",
        "Center E"
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


# Bar Chart
plt.bar(df["name"], df["water_usage"])

plt.title("Data Center Water Usage")
plt.xlabel("Data Center")
plt.ylabel("Water Usage")

plt.show()