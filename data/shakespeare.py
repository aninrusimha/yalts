import os
import requests
import tiktoken
import numpy as np

# download the tiny shakespeare dataset
input_file_path = os.path.join(os.path.dirname(__file__), "input.txt")
if not os.path.exists(input_file_path):
    data_url = "https://www.gutenberg.org/cache/epub/100/pg100.txt"
    with open(input_file_path, "w", encoding="utf-8") as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, "r", encoding="utf-8") as f:
    data = f.read()
data = data.split("\n\n")
data = [line.strip() for line in data if line.strip() and "\n" in line]
for i in range(1, 9):
    print(data[i * 100])
n = len(data)
np.random.shuffle(data)
train_data = data[: int(n * 0.95)]
val_data = data[int(n * 0.95) :]
SL = 2048

# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
# tokenization matters! How could we improve this to make more sense?
train_ids = [[enc.eot_token] + enc.encode_ordinary(ex) for ex in train_data]
print(max(len(line) for line in train_ids))
print(min(len(line) for line in train_ids))
val_ids = [[enc.eot_token] + enc.encode_ordinary(ex) for ex in val_data]
print(f"train has {sum([len(ex) for ex in train_ids]):,} tokens")
print(f"val has {sum([len(ex) for ex in val_ids]):,} tokens")

# pack into sequences of length SL

train_ids_packed = []
cur_example = []
for i in range(0, len(train_ids)):
    cur_example += train_ids[i]
    if len(cur_example) >= SL:
        train_ids_packed.append(cur_example[:SL])
        cur_example = cur_example[SL:]
val_ids_packed = []
cur_example = []
for i in range(0, len(val_ids)):
    cur_example += val_ids[i]
    if len(cur_example) >= SL:
        val_ids_packed.append(cur_example[:SL])
        cur_example = cur_example[SL:]
print(f"train.bin has {len(train_ids_packed):,} examples")
print(f"val.bin has {len(val_ids_packed):,} examples")
# export to bin files
train_ids = np.array(train_ids_packed, dtype=np.uint16)
val_ids = np.array(val_ids_packed, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), "train.bin"))
val_ids.tofile(os.path.join(os.path.dirname(__file__), "val.bin"))

# train.bin has 301,966 tokens
# val.bin has 36,059 tokens
