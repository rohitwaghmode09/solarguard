import os
import matplotlib.pyplot as plt

classes = [
    "Bird-drop",
    "Clean",
    "Dusty",
    "Electrical-damage",
    "Physical-Damage",
    "Snow-Covered"
]

image_counts = []

for class_name in classes:
    folder = class_name

    images = [
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    image_counts.append(len(images))

plt.figure(figsize=(10, 6))
plt.bar(classes, image_counts)

plt.title("Solar Panel Dataset - Class Distribution")
plt.xlabel("Fault / Condition")
plt.ylabel("Number of Images")

plt.xticks(rotation=30)
plt.tight_layout()

plt.show()