import os
import random
import datasets
import evaluate
import threading
from write_result import write_jsonl, read_jsonl
from predict import deepseek_predict, gpt_predict
from datasets import load_dataset
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

data = read_jsonl("deepseek_r1_results/MNLI_3.jsonl")
predictions = []
references = []
for line in data:
    predictions.append(line["prediction"])
    references.append(line["label"])

accuracy = accuracy_metric.compute(predictions=predictions, references=references)
f1 = f1_metric.compute(predictions=predictions, references=references, average="macro")


print(f"Accuracy: {accuracy['accuracy']}")
print(f"F1      : {f1['f1']}")
