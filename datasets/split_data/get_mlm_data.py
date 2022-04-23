import json
import random


in_path = ["train.json", "test.json", "dev.json"]
train_out = "mlm/mlm_train.txt"
dev_out = "mlm/mlm_dev.txt"

out_train = open(train_out, "w")
out_dev = open(dev_out, "w")

for i in in_path:
    instances = json.load(open(i, 'r'))
    for ins in instances:
        if random.random() > 0.05:
            out_train.write(ins[0] + "\n")
        else:
            out_dev.write(ins[0] + "\n")
