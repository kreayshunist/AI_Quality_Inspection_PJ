import os
import csv

base_dir = "data/hawk_quality"
output_csv = "data/patchcore_data/label.csv"


label_map = {
    "good": 0,
    "bad": 1
}

rows = []

for label_name, label_value in label_map.items():
    folder_path = os.path.join(base_dir, label_name)
    if not os.path.isdir(folder_path):
        continue  
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".jpg"):
            rows.append([filename, label_value])


os.makedirs(os.path.dirname(output_csv), exist_ok=True)
with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "label"])
    writer.writerows(rows)

print(f"Saved to: {output_csv}")
