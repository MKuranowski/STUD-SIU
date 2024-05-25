import csv

filename = 'dqn_multi_models.csv'

data = []
with open(filename, mode='r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        reward = float(row['reward'])
        hash_value = row['hash']
        signature = row['signature']
        data.append((reward, hash_value, signature))

data.sort(reverse=True, key=lambda x: x[0])

with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['reward', 'hash', 'signature'])
    for row in data:
        writer.writerow(row)

print(f"Dane zostały posortowane i zapisane w pliku '{filename}'.")
