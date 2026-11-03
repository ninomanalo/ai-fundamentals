name = "Nino"
age = 20
is_learning_ai = True

print(name)
print(age)
print(is_learning_ai)

temperatures = [30.5, 31.2, 29.8, 32.1, 33.0]

print(temperatures)
print(temperatures[0])
print(len(temperatures))


data_center = {
    "company": "AWS",
    "country": "USA",
    "water_usage": 1450.7
}

print(data_center["company"])
print(data_center["water_usage"])


water_usage = 1500

if water_usage > 1000:
    print("High water usage")
else:
    print("Normal water usage")


water_usage = [900, 1200, 1500, 800, 1800]

for usage in water_usage:
    print(usage)


for usage in water_usage:
    if usage > 1000:
        print("High:", usage)
    else:
        print("Normal:", usage)


def check_water_usage(usage):
    if usage > 1000:
        return "High"
    else:
        return "Normal"


print(check_water_usage(1500))
print(check_water_usage(800))


data_centers = [
    {"name": "Center A", "water_usage": 850},
    {"name": "Center B", "water_usage": 1200},
    {"name": "Center C", "water_usage": 950},
    {"name": "Center D", "water_usage": 1800},
    {"name": "Center E", "water_usage": 1350}
]


def classify_water_usage(usage):
    if usage > 1000:
        return "High"
    else:
        return "Normal"


total_usage = 0

for center in data_centers:
    name = center["name"]
    usage = center["water_usage"]

    classification = classify_water_usage(usage)

    print(f"{name} - {usage} - {classification}")

    total_usage += usage


average_usage = total_usage / len(data_centers)

print()
print(f"Average Water Usage: {average_usage}")