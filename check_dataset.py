import os

classes = [
    "Bird-drop",
    "Clean",
    "Dusty",
    "Electrical-damage",
    "Physical-Damage",
    "Snow-Covered"
]

for class_name in classes:
    folder = class_name

    if os.path.exists(folder):
        images = [
            f for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        print(f"{class_name}: {len(images)} images")
    else:
        print(f"{class_name}: FOLDER NOT FOUND")

        