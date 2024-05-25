import csv
import os

data = []
with open("dqn_multi_models.csv", mode="r") as file:
    reader = csv.DictReader(file)
    for row in reader:
        reward = float(row["reward"])
        hash_value = row["hash"]
        signature = row["signature"]
        data.append((reward, hash_value, signature))

for reward, hash_value, signature in data:
    if reward < 2000 and hash_value != "2d4f0f":
        filename = f"dqnm-{hash_value}-{signature}.h5"
        if os.path.exists("./models/" + filename):
            os.remove("models/" + filename)
            print(f"Plik {filename} został usunięty.")
