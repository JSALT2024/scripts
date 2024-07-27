import json
import random


def split_dict(input_dict, split_ratio=0.9):
    # Convert the dictionary items to a list
    items = list(input_dict.items())

    # Determine the split point
    split_point = int(len(items) * split_ratio)

    # Split the items into two parts
    dict1_items = items[:split_point]
    dict2_items = items[split_point:]

    # Create new dictionaries from the split items
    dict1 = dict(dict1_items)
    dict2 = dict(dict2_items)

    return dict1, dict2


with open(r'C:\Work\JSALT\scripts\data\ytasl\YT.annotations.train.new.json') as f:
    data = json.load(f)

validation_split = 0.1
num_val = int(len(data) * validation_split)
train_data, val_data = split_dict(data, split_ratio=0.9)

with open(r'C:\Work\JSALT\scripts\data\ytasl\YT.annotations.train.json', 'w') as f:
    json.dump(train_data, f)

with open(r'C:\Work\JSALT\scripts\data\ytasl\YT.annotations.val.json', 'w') as f:
    json.dump(val_data, f)
