import os
import random
import datasets
import evaluate
import threading
from write_result import write_jsonl
from predict import deepseek_predict
from datasets import load_dataset
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('result_dir', type=str)
parser.add_argument('num_samples', type=int, required=True)
parser.add_argument('batch_size', type=int, default=1)
parser.add_argument('api_key', type=str)
args = parser.parse_args()

if args.result_dir:
    result_dir = args.result_dir
else:
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    result_dir = os.path.join(script_dir, 'deepseek_r1_results')
num_samples = args.num_samples
batch_size = args.batch_size
api_key = args.api_key
dataset = load_dataset("facebook/anli")


indices = random.sample(range(len(dataset)), num_samples)
eval_subset = dataset.select(indices)


accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")
print("Dataset: ANLI")



def get_model_prediction(premise, hypothesis, label) -> int:
    system_message = '''You are tasked with determining the logical relationship between two sentences. There are three possible relationships:

1. Entailment: The second sentence logically follows from the first.
2. Neutral: The second sentence is unrelated or has no clear connection to the first.
3. Contradiction: The second sentence conflicts with the first.

For each pair of sentences, classify their relationship as "Entailment", "Neutral", or "Contradiction". Don't say anything else.
'''
    prediction = deepseek_predict(system_message, "Premise: "+premise+ "\nHypothesis: "+hypothesis, api_key=api_key)

    if prediction[0:10] == "Entailment":
        pred_label = 0
    elif prediction[0:7] == "Neutral":
        pred_label = 1
    elif prediction[0:13] == "Contradiction":
        pred_label = 2
    else:
        pred_label = 0

    predictions.append(pred_label)
    references.append(label)
    entry = {}
    entry["sentence"] = {"premise": premise, "hypothesis":hypothesis}
    entry["label"] = label
    entry["prediction"] = pred_label
    entry["original_prediction"] = prediction
    data_entries.append(entry)


predictions = []
references = []
threads = []
data_entries = []


for i in range(0, len(eval_subset), batch_size):
    batch = eval_subset[i:i + batch_size]
    for j in range(len(batch["premise"])):

        premise = batch["premise"][j]
        hypothesis = batch["hypothesis"][j]
        label = batch["label"][j]
        thread = threading.Thread(target=get_model_prediction, args=(premise,hypothesis, label,))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()




accuracy = accuracy_metric.compute(predictions=predictions, references=references)
f1 = f1_metric.compute(predictions=predictions, references=references, average="macro")


print(f"Accuracy: {accuracy['accuracy']:.4f}")
print(f"F1      : {f1['f1']:.4f}")
results = {"model":"deepseek-r1","dataset":"ANLI","num_samples":num_samples,"accuracy": accuracy["accuracy"],"F1":f1['f1']}
write_jsonl( os.path.join(result_dir, "overall_results.jsonl"),[results])
write_jsonl( os.path.join(result_dir, "ANLI.jsonl"),data_entries, 'w')
