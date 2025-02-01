import os
import random
import datasets
import evaluate
import threading
from write_result import write_jsonl
from predict import deepseek_predict
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


dataset = datasets.load_dataset("glue", "sst2", split="validation")


indices = random.sample(range(len(dataset)), num_samples)
eval_subset = dataset.select(indices)


accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")



def get_model_prediction(input_text: str, label) -> int:
    system_message = '''You are a sentiment analysis expert. Classify the sentiment of the following sentence as either positive or negative based on its tone and context.

Respond with only one word: "Positive" or "Negative".'''
    prediction = deepseek_predict(system_message, "Sentence: "+input_text, api_key=api_key)
    pred_label = 1 if prediction== "Positive" else 0

    predictions.append(pred_label)
    references.append(label)
    entry = {}
    entry["sentence"] = input_text
    entry["label"] = label
    entry["prediction"] = pred_label
    data_entries.append(entry)


predictions = []
references = []
threads = []
data_entries = []

for i in range(0, len(eval_subset), batch_size):
    batch = eval_subset[i:i + batch_size]
    for j in range(len(batch["sentence"])):

        text = batch["sentence"][j]
        label = batch["label"][j]  # 0 表示消极，1 表示积极
        thread = threading.Thread(target=get_model_prediction, args=(text, label,))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()
    print(f"Batch {i+1}/{len(eval_subset)} Processed.")




accuracy = accuracy_metric.compute(predictions=predictions, references=references)
f1 = f1_metric.compute(predictions=predictions, references=references)


print(f"Accuracy: {accuracy['accuracy']:.4f}")
print(f"F1      : {f1['f1']:.4f}")
results = {"model":"deepseek-r1","dataset":"SST-2","num_samples":num_samples,"accuracy": accuracy["accuracy"],"F1":f1['f1']}
write_jsonl( os.path.join(result_dir, "overall_results.jsonl"),[results])
write_jsonl( os.path.join(result_dir, "SST-2.jsonl"),data_entries, 'w')
