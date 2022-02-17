import re
import string

import pandas as pd
import numpy as np
from tqdm import tqdm
from datasets import Dataset

BASE_DIR = "/root/kaggle/feedback-prize-2021/data"

df = pd.read_csv(f"{BASE_DIR}/train.csv")

# for each row, grab the span of text from the file using discourse_start and discourse_end
def get_text_by_index(example):
    id_ = example["id"]
    start = example["discourse_start"]
    end = example["discourse_end"]
    with open(f"{BASE_DIR}/train/{id_}.txt") as fp:
        file_text = fp.read()
    return {
        "text_by_index": file_text[int(start) : int(end)]
    }

id_ds = Dataset.from_pandas(df[["id", "discourse_start", "discourse_end"]])

text_ds = id_ds.map(get_text_by_index, num_proc=4)
df["text_by_index"] = text_ds["text_by_index"]

not_equal_texts = df[df["discourse_text"] != df["text_by_index"]]
print(f"There are {len(not_equal_texts)} that are not equal")

# Let's look at a few
discourse_texts = not_equal_texts["discourse_text"]
file_spans = not_equal_texts["text_by_index"]
discourse_ids = not_equal_texts["discourse_id"]

for counter, (discourse_text, file_span, discourse_id) in enumerate(
    zip(discourse_texts, file_spans, discourse_ids)
):
    if counter > 5:
        break

    if len(discourse_text) != len(file_span):
        continue

    print("discourse_id =", discourse_id)
    print("\n***discourse_text in train.csv***\n")
    print(discourse_text)
    print("\n"+"-" * 20)
    print("\n***Using discourse_start and discourse_end***\n")
    print(file_span)

    # Print index of character that differs between the two texts
    print(
        [
            (i, char1, char2)
            for i, (char1, char2) in enumerate(zip(discourse_text, file_span))
            if char1 != char2
        ]
    )

    print("\n" + "*" * 20 + "\n")

counter = 0
discourse_texts = not_equal_texts["discourse_text"]
file_spans = not_equal_texts["text_by_index"]
discourse_ids = not_equal_texts["discourse_id"]

for discourse_text, file_span, discourse_id in zip(
    discourse_texts, file_spans, discourse_ids
):
    if counter >= 2:
        break

    if len(discourse_text) != len(file_span):
        continue

    # Print index of character that differs between the two texts
    diffs = [
        (i, char1, char2)
        for i, (char1, char2) in enumerate(zip(discourse_text, file_span))
        if char1 != char2
    ]

    if not diffs[0][1].isalpha():
        continue

    print("discourse_id =", discourse_id)
    print("\n***discourse_text in train.csv***\n")
    print(discourse_text)
    print("-" * 20)
    print("\n***Using discourse_start and discourse_end***\n")
    print(file_span)

    # Print index of difference in char
    print(diffs)

    print("\n" + "*" * 20 + "\n")
    counter += 1

from collections import Counter

all_diffs = []
for discourse_text, file_text in not_equal_texts[["discourse_text", "text_by_index"]].values:
    
    if len(discourse_text) != len(file_text):
        continue
        
    all_diffs.extend([(char1, char2) for char1, char2 in zip(discourse_text, file_text) if char1!=char2])

    
counter = Counter(all_diffs)

counter.most_common(20)