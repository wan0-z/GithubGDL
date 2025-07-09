import random
from pathlib import Path
import os


def write_txt(file_path, data):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    f = open(file_path, "w")
    for d in data:
        f.write(str(d.stem) + '\n')
    f.close()


data_path = Path("../data/graphs")
graph_path = list(data_path.glob("20250701*.json"))
random.shuffle(graph_path)

data_size = len(graph_path)
train_size = int(data_size * 0.7)
val_size = int(data_size * 0.15)
test_size = data_size - train_size - val_size

train_graphs = graph_path[0:train_size]
val_graphs = graph_path[train_size:train_size + val_size]
test_graphs = graph_path[train_size + val_size:]

assert len(train_graphs) + len(val_graphs) + len(test_graphs) == data_size

write_txt("../data/partition/train.txt", train_graphs)
write_txt("../data/partition/val.txt", val_graphs)
write_txt("../data/partition/test.txt", test_graphs)
