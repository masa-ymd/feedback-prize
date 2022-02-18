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

PUNCTUATION = set(".,;")

def get_new_positions(examples):
    
    disc_ids = []
    new_starts = []
    new_ends = []
    new_texts = []
    
    for id_ in examples["id"]:
    
        with open(f"{BASE_DIR}/train/{id_}.txt") as fp:
            file_text = fp.read()

        discourse_data = df[df["id"] == id_]

        discourse_ids = discourse_data["discourse_id"]
        discourse_texts = discourse_data["discourse_text"]
        discourse_starts = discourse_data["discourse_start"]
        for disc_id, disc_text, disc_start in zip(discourse_ids, discourse_texts, discourse_starts):
            disc_text = disc_text.strip()

            matches = [x for x in re.finditer(re.escape(disc_text), file_text)]
            offset = 0
            while len(matches) == 0 and offset < len(disc_text):
                chunk = disc_text if offset == 0 else disc_text[:-offset]
                matches = [x for x in re.finditer(re.escape(chunk), file_text)]
                offset += 5
            if offset >= len(disc_text):
                print(f"Could not find substring in {disc_id}")
                continue

            # There are some instances when there are multiple matches, 
            # so we'll take the closest one to the original discourse_start
            distances = [abs(disc_start-match.start()) for match in matches]

            idx = matches[np.argmin(distances)].start()                

            end_idx = idx + len(disc_text)

            # if it starts with whitespace or punctuation, increase idx
            while file_text[idx].split()==[] or file_text[idx] in PUNCTUATION:
                idx += 1
            
            # if the next 
            if (end_idx < len(file_text) and 
                (file_text[end_idx-1]!=[] or file_text[end_idx-1] not in PUNCTUATION) and 
                (file_text[end_idx].split()==[] or file_text[end_idx] in PUNCTUATION)):
                end_idx += 1

            final_text = file_text[idx:end_idx]
            
            disc_ids.append(disc_id)
            new_starts.append(idx)
            new_ends.append(idx + len(final_text))
            new_texts.append(final_text)
            
    return {
        "discourse_id": disc_ids,
        "new_start": new_starts,
        "new_end": new_ends,
        "text_by_new_index": new_texts,
    }

# using Dataset will make it easy to do multi-processing        
dataset = Dataset.from_dict({"id": df["id"].unique()})   

results = dataset.map(get_new_positions, batched=True, num_proc=4, remove_columns=["id"])

df["new_start"] = results["new_start"]
df["new_end"] = results["new_end"]
df["text_by_new_index"] = results["text_by_new_index"]

new_not_equal_texts = df[df["discourse_text"]!=df["text_by_new_index"]].copy()
print(f"There are {new_not_equal_texts['id'].nunique()} files and {len(new_not_equal_texts)} rows with mismatched spans.")

new_not_equal_texts["discourse_text"] = new_not_equal_texts["discourse_text"]
new_not_equal_texts["text_by_new_index"] = new_not_equal_texts["text_by_new_index"]

# if we cutoff the last few characters, they will are more likely to be equal
old_text = new_not_equal_texts["discourse_text"].str.strip().str.slice(start=2, stop=3)
new_text = new_not_equal_texts["text_by_new_index"].str.strip().str.slice(start=2, stop=3)


char_unequal_mask = old_text!=new_text

unequal_texts = new_not_equal_texts[char_unequal_mask]

unequal_texts[["discourse_text", "text_by_new_index"]].sample(n=25).values

def find_pred_string(examples):
    
    new_pred_strings = []
    discourse_ids = []
    
    for id_ in examples["id"]:
        with open(f"{BASE_DIR}/train/{id_}.txt") as fp:
            file_text = fp.read()

        discourse_data = df[df["id"] == id_]
        
        left_idxs = discourse_data["new_start"]
        right_idxs = discourse_data["new_end"]
        disc_ids = discourse_data["discourse_id"]
        
        for left_idx, right_idx, disc_id in zip(left_idxs, right_idxs, disc_ids):
            start_word_id = len(file_text[:left_idx].split())
            
            # In the event that the first character of the span is not whitespace
            # and the character before the span is not whitespace, `len(span.split())`
            # will need to be reduced by 1.
            # ex: word__word___sp[an starts in the middle of a word]
            # `len(text[:left_idx].split())==3` but it actually starts in the 3rd word 
            # which is word_id=2
            if left_idx > 0 and file_text[left_idx].split() != [] and file_text[left_idx-1].split() != []:
                start_word_id -= 1
                
            end_word_id = start_word_id + len(file_text[left_idx:right_idx].split())
            
            new_pred_strings.append(" ".join(list(map(str, range(start_word_id, end_word_id)))))
            discourse_ids.append(disc_id)
            
            
    return {
        "new_predictionstring": new_pred_strings,
        "discourse_id": discourse_ids
    }
        

id_ds = Dataset.from_pandas(df[["id"]].drop_duplicates())
new_pred_string_ds = id_ds.map(find_pred_string, batched=True, num_proc=4, remove_columns=id_ds.column_names)

df["new_predictionstring"] = new_pred_string_ds["new_predictionstring"]
len([x for x in new_pred_string_ds["new_predictionstring"] if x == ""])

different_value_mask = df["new_predictionstring"] != df["predictionstring"]

for idx, row in df[different_value_mask].sample(n=5, random_state=18).iterrows():
    file_text = open(f"{BASE_DIR}/train/{row.id}.txt").read()
    print("Old predictionstring=", row.predictionstring)
    print("New predictionstring=", row.new_predictionstring)
    print("words using old predictionstring=", [x for i, x in enumerate(file_text.split()) if i in list(map(int, row.predictionstring.split()))])
    print("words using new predictionstring=", [x for i, x in enumerate(file_text.split()) if i in list(map(int, row.new_predictionstring.split()))])
    print("discourse text=", row.text_by_new_index)
    print(f"start_idx/end_idx= {row.new_start}/{row.new_end}")
    print("discourse_id=",row.discourse_id, "\n")

print(sum(df["discourse_start"].astype(int) != df["new_start"]))
print(sum(df["discourse_end"].astype(int) != df["new_end"]))

df.to_csv(f"{BASE_DIR}/corrected_train.csv", index=False)