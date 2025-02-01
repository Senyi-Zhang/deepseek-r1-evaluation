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

dataset = load_dataset("maxmoynan/SemEval2017-Task4aEnglish")

indices = random.sample(range(len(dataset)), num_samples)
eval_subset = dataset.select(indices)


accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")
print("Dataset: SemEval2017-Task4")



def get_model_prediction(input_text: str, label) -> int:
    system_message = '''You are a sentiment analysis model trained on Twitter data. Your task is to classify the sentiment of the given tweet into one of the following categories:
- "negative" (0): The tweet expresses negative sentiment, such as criticism, frustration, anger, or disappointment.
- "neutral" (1): The tweet is neutral, factual, or does not express emotions.
- "positive" (2): The tweet expresses positive sentiment, such as praise, happiness, excitement, or enthusiasm.


Your response should be a single number: 
- 0 for negative
- 1 for neutral
- 2 for positive
'''
    prediction = deepseek_predict(system_message, "Tweet: "+input_text, api_key=api_key)
    print(input_text)
    print(prediction)
    try:
        pred_label = int(prediction)
    except ValueError:
        pred_label = 1

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

    for j in range(len(batch["tweet"])):

        text = batch["tweet"][j]
        label = batch["sentiment"][j]
        thread = threading.Thread(target=get_model_prediction, args=(text, label,))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()
    print(f"Batch {i+1}/{len(eval_subset)} Processed.")




accuracy = accuracy_metric.compute(predictions=predictions, references=references)
f1 = f1_metric.compute(predictions=predictions, references=references, average="macro")

print("======= Evaluation Results  =======")
print(f"Accuracy: {accuracy['accuracy']:.4f}")
print(f"F1      : {f1['f1']:.4f}")
results = {"model":"deepseek=r1","dataset":"SemEval2017t4","num_samples":num_samples,"accuracy": accuracy["accuracy"],"F1":f1['f1']}
write_jsonl( os.path.join(result_dir, "overall_results.jsonl"),[results])
write_jsonl( os.path.join(result_dir, "SemEval2017t4.jsonl"),data_entries, 'w')
