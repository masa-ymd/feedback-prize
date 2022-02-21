SAMPLE = True # set True for debugging

import wandb




wandb.login()
wandb.init(project="feedback-prize-2021")


# CONFIG

EXP_NUM = 4
task = "ner"
model_checkpoint = "allenai/longformer-base-4096"
max_length = 1024
stride = 128
min_tokens = 6
model_path = f'{model_checkpoint.split("/")[-1]}-{EXP_NUM}'

# TRAINING HYPERPARAMS
BS = 4
GRAD_ACC = 8
LR = 5e-5
WD = 0.01
WARMUP = 0.1
N_EPOCHS = 5

import pandas as pd

# read train data
train = pd.read_csv('../input/feedback-prize-2021/train.csv')
train.head(1)

# check unique classes
classes = train.discourse_type.unique().tolist()
print(classes)


# setup label indices

from collections import defaultdict
tags = defaultdict()

for i, c in enumerate(classes):
    tags[f'B-{c}'] = i
    tags[f'I-{c}'] = i + len(classes)
tags[f'O'] = len(classes) * 2
tags[f'Special'] = -100
    
l2i = dict(tags)

i2l = defaultdict()
for k, v in l2i.items(): 
    i2l[v] = k
i2l[-100] = 'Special'

i2l = dict(i2l)

N_LABELS = len(i2l) - 1 # not accounting for -100

# some helper functions

from pathlib import Path

path = Path('../input/feedback-prize-2021/train')

def get_raw_text(ids):
    with open(path/f'{ids}.txt', 'r') as file: data = file.read()
    return data

# group training labels by text file

df1 = train.groupby('id')['discourse_type'].apply(list).reset_index(name='classlist')
df2 = train.groupby('id')['discourse_start'].apply(list).reset_index(name='starts')
df3 = train.groupby('id')['discourse_end'].apply(list).reset_index(name='ends')
df4 = train.groupby('id')['predictionstring'].apply(list).reset_index(name='predictionstrings')

df = pd.merge(df1, df2, how='inner', on='id')
df = pd.merge(df, df3, how='inner', on='id')
df = pd.merge(df, df4, how='inner', on='id')
df['text'] = df['id'].apply(get_raw_text)

print(df.head())

# debugging
if SAMPLE: df = df.sample(n=100).reset_index(drop=True)

# we will use HuggingFace datasets
from datasets import Dataset, load_metric

ds = Dataset.from_pandas(df)
datasets = ds.train_test_split(test_size=0.1, shuffle=True, seed=42)
datasets

from transformers import AutoTokenizer
    
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)

# Not sure if this is needed, but in case we create a span with certain class without starting token of that class,
# let's convert the first token to be the starting token.

e = [0,7,7,7,1,1,8,8,8,9,9,9,14,4,4,4]

def fix_beginnings(labels):
    for i in range(1,len(labels)):
        curr_lab = labels[i]
        prev_lab = labels[i-1]
        if curr_lab in range(7,14):
            if prev_lab != curr_lab and prev_lab != curr_lab - 7:
                labels[i] = curr_lab -7
    return labels

fix_beginnings(e)

# tokenize and add labels
def tokenize_and_align_labels(examples):

    o = tokenizer(examples['text'], truncation=True, padding=True, return_offsets_mapping=True, max_length=max_length, stride=stride, return_overflowing_tokens=True)

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = o["overflow_to_sample_mapping"]
    # The offset mappings will give us a map from token to character position in the original context. This will
    # help us compute the start_positions and end_positions.
    offset_mapping = o["offset_mapping"]
    
    o["labels"] = []

    for i in range(len(offset_mapping)):
                   
        sample_index = sample_mapping[i]

        labels = [l2i['O'] for i in range(len(o['input_ids'][i]))]

        for label_start, label_end, label in \
        list(zip(examples['starts'][sample_index], examples['ends'][sample_index], examples['classlist'][sample_index])):
            for j in range(len(labels)):
                token_start = offset_mapping[i][j][0]
                token_end = offset_mapping[i][j][1]
                if token_start == label_start: 
                    labels[j] = l2i[f'B-{label}']    
                if token_start > label_start and token_end <= label_end: 
                    labels[j] = l2i[f'I-{label}']

        for k, input_id in enumerate(o['input_ids'][i]):
            if input_id in [0,1,2]:
                labels[k] = -100

        labels = fix_beginnings(labels)
                   
        o["labels"].append(labels)
        
    return o

tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=True, \
                                  batch_size=20000, remove_columns=datasets["train"].column_names)

print(tokenized_datasets)

# we will use auto model for token classification

from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer

model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=N_LABELS)

model_name = model_checkpoint.split("/")[-1]
args = TrainingArguments(
    f"{model_name}-finetuned-{task}",
    evaluation_strategy = "epoch",
    logging_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=LR,
    per_device_train_batch_size=BS,
    per_device_eval_batch_size=BS,
    num_train_epochs=N_EPOCHS,
    weight_decay=WD,
    report_to='wandb', 
    gradient_accumulation_steps=GRAD_ACC,
    warmup_ratio=WARMUP
)

from transformers import DataCollatorForTokenClassification

data_collator = DataCollatorForTokenClassification(tokenizer)

# this is not the competition metric, but for now this will be better than nothing...

metric = load_metric("seqeval")

import numpy as np

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [i2l[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [i2l[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics, 
)

trainer.train()


trainer.save_model(model_path)

def tokenize_for_validation(examples):

    o = tokenizer(examples['text'], truncation=True, return_offsets_mapping=True, max_length=4096)

    # The offset mappings will give us a map from token to character position in the original context. This will
    # help us compute the start_positions and end_positions.
    offset_mapping = o["offset_mapping"]
    
    o["labels"] = []

    for i in range(len(offset_mapping)):
                   
        labels = [l2i['O'] for i in range(len(o['input_ids'][i]))]

        for label_start, label_end, label in \
        list(zip(examples['starts'][i], examples['ends'][i], examples['classlist'][i])):
            for j in range(len(labels)):
                token_start = offset_mapping[i][j][0]
                token_end = offset_mapping[i][j][1]
                if token_start == label_start: 
                    labels[j] = l2i[f'B-{label}']    
                if token_start > label_start and token_end <= label_end: 
                    labels[j] = l2i[f'I-{label}']

        for k, input_id in enumerate(o['input_ids'][i]):
            if input_id in [0,1,2]:
                labels[k] = -100

        labels = fix_beginnings(labels)
                   
        o["labels"].append(labels)
        
    return o

tokenized_val = datasets.map(tokenize_for_validation, batched=True)
tokenized_val

# ground truth for validation

l = []
for example in tokenized_val['test']:
    for c, p in list(zip(example['classlist'], example['predictionstrings'])):
        l.append({
            'id': example['id'],
            'discourse_type': c,
            'predictionstring': p,
        })
    
gt_df = pd.DataFrame(l)
print(gt_df)

# visualization with displacy

import pandas as pd
import os
from pathlib import Path
import spacy
from spacy import displacy
from pylab import cm, matplotlib

path = Path('../input/feedback-prize-2021/train')

colors = {
            'Lead': '#8000ff',
            'Position': '#2b7ff6',
            'Evidence': '#2adddd',
            'Claim': '#80ffb4',
            'Concluding Statement': 'd4dd80',
            'Counterclaim': '#ff8042',
            'Rebuttal': '#ff0000',
            'Other': '#007f00',
         }

def visualize(df, text):
    ents = []
    example = df['id'].loc[0]

    for i, row in df.iterrows():
        ents.append({
                        'start': int(row['discourse_start']), 
                         'end': int(row['discourse_end']), 
                         'label': row['discourse_type']
                    })

    doc2 = {
        "text": text,
        "ents": ents,
        "title": example
    }

    options = {"ents": train.discourse_type.unique().tolist() + ['Other'], "colors": colors}
    displacy.render(doc2, style="ent", options=options, manual=True, jupyter=True)
predictions, labels, _ = trainer.predict(tokenized_val['test'])

preds = np.argmax(predictions, axis=-1)
print(preds.shape)

# code that will convert our predictions into prediction strings, and visualize it at the same time
# this most likely requires some refactoring

def get_class(c):
    if c == 14: return 'Other'
    else: return i2l[c][2:]

def pred2span(pred, example, viz=False, test=False):
    example_id = example['id']
    n_tokens = len(example['input_ids'])
    classes = []
    all_span = []
    for i, c in enumerate(pred.tolist()):
        if i == n_tokens-1:
            break
        if i == 0:
            cur_span = example['offset_mapping'][i]
            classes.append(get_class(c))
        elif i > 0 and (c == pred[i-1] or (c-7) == pred[i-1]):
            cur_span[1] = example['offset_mapping'][i][1]
        else:
            all_span.append(cur_span)
            cur_span = example['offset_mapping'][i]
            classes.append(get_class(c))
    all_span.append(cur_span)
    
    if test: text = get_test_text(example_id)
    else: text = get_raw_text(example_id)
    
    # abra ka dabra se soli fanta ko pelo
    
    # map token ids to word (whitespace) token ids
    predstrings = []
    for span in all_span:
        span_start = span[0]
        span_end = span[1]
        before = text[:span_start]
        token_start = len(before.split())
        if len(before) == 0: token_start = 0
        elif before[-1] != ' ': token_start -= 1
        num_tkns = len(text[span_start:span_end+1].split())
        tkns = [str(x) for x in range(token_start, token_start+num_tkns)]
        predstring = ' '.join(tkns)
        predstrings.append(predstring)
                    
    rows = []
    for c, span, predstring in zip(classes, all_span, predstrings):
        e = {
            'id': example_id,
            'discourse_type': c,
            'predictionstring': predstring,
            'discourse_start': span[0],
            'discourse_end': span[1],
            'discourse': text[span[0]:span[1]+1]
        }
        rows.append(e)


    df = pd.DataFrame(rows)
    df['length'] = df['discourse'].apply(lambda t: len(t.split()))
    
    # short spans are likely to be false positives, we can choose a min number of tokens based on validation
    df = df[df.length > min_tokens].reset_index(drop=True)
    if viz: visualize(df, text)

    return df
pred2span(preds[0], tokenized_val['test'][0], viz=True)


# source: https://www.kaggle.com/robikscube/student-writing-competition-twitch#Competition-Metric-Code

def calc_overlap(row):
    """
    Calculates the overlap between prediction and
    ground truth and overlap percentages used for determining
    true positives.
    """
    set_pred = set(row.predictionstring_pred.split(" "))
    set_gt = set(row.predictionstring_gt.split(" "))
    # Length of each and intersection
    len_gt = len(set_gt)
    len_pred = len(set_pred)
    inter = len(set_gt.intersection(set_pred))
    overlap_1 = inter / len_gt
    overlap_2 = inter / len_pred
    return [overlap_1, overlap_2]


def score_feedback_comp_micro(pred_df, gt_df):
    """
    A function that scores for the kaggle
        Student Writing Competition

    Uses the steps in the evaluation page here:
        https://www.kaggle.com/c/feedback-prize-2021/overview/evaluation
    """
    gt_df = (
        gt_df[["id", "discourse_type", "predictionstring"]]
        .reset_index(drop=True)
        .copy()
    )
    pred_df = pred_df[["id", "class", "predictionstring"]].reset_index(drop=True).copy()
    pred_df["pred_id"] = pred_df.index
    gt_df["gt_id"] = gt_df.index
    # Step 1. all ground truths and predictions for a given class are compared.
    joined = pred_df.merge(
        gt_df,
        left_on=["id", "class"],
        right_on=["id", "discourse_type"],
        how="outer",
        suffixes=("_pred", "_gt"),
    )
    joined["predictionstring_gt"] = joined["predictionstring_gt"].fillna(" ")
    joined["predictionstring_pred"] = joined["predictionstring_pred"].fillna(" ")

    joined["overlaps"] = joined.apply(calc_overlap, axis=1)

    # 2. If the overlap between the ground truth and prediction is >= 0.5,
    # and the overlap between the prediction and the ground truth >= 0.5,
    # the prediction is a match and considered a true positive.
    # If multiple matches exist, the match with the highest pair of overlaps is taken.
    joined["overlap1"] = joined["overlaps"].apply(lambda x: eval(str(x))[0])
    joined["overlap2"] = joined["overlaps"].apply(lambda x: eval(str(x))[1])

    joined["potential_TP"] = (joined["overlap1"] >= 0.5) & (joined["overlap2"] >= 0.5)
    joined["max_overlap"] = joined[["overlap1", "overlap2"]].max(axis=1)
    tp_pred_ids = (
        joined.query("potential_TP")
        .sort_values("max_overlap", ascending=False)
        .groupby(["id", "predictionstring_gt"])
        .first()["pred_id"]
        .values
    )

    # 3. Any unmatched ground truths are false negatives
    # and any unmatched predictions are false positives.
    fp_pred_ids = [p for p in joined["pred_id"].unique() if p not in tp_pred_ids]

    matched_gt_ids = joined.query("potential_TP")["gt_id"].unique()
    unmatched_gt_ids = [c for c in joined["gt_id"].unique() if c not in matched_gt_ids]

    # Get numbers of each type
    TP = len(tp_pred_ids)
    FP = len(fp_pred_ids)
    FN = len(unmatched_gt_ids)
    # calc microf1
    my_f1_score = TP / (TP + 0.5 * (FP + FN))
    return my_f1_score


def score_feedback_comp(pred_df, gt_df, return_class_scores=False):
    class_scores = {}
    pred_df = pred_df[["id", "class", "predictionstring"]].reset_index(drop=True).copy()
    for discourse_type, gt_subset in gt_df.groupby("discourse_type"):
        pred_subset = (
            pred_df.loc[pred_df["class"] == discourse_type]
            .reset_index(drop=True)
            .copy()
        )
        class_score = score_feedback_comp_micro(pred_subset, gt_subset)
        class_scores[discourse_type] = class_score
    f1 = np.mean([v for v in class_scores.values()])
    if return_class_scores:
        return f1, class_scores
    return f1

print(score_feedback_comp(pred_df, gt_df, return_class_scores=True))
wandb.log({'cv': score_feedback_comp(pred_df, gt_df, return_class_scores=False)})
wandb.finish()