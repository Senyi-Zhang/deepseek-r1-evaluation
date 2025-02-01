# deepseek-r1-evaluation
Evaluation code for deepseek-r1 and other llms.
## Introduction
This code is designed to test deepseek-r1(while useful for other models as well) on six common NLP datasets(SST-2, SemEval2017task4, MNLI, ANLI, CommonsenseQA, TruthfulQA)
## Start
First, run
```
pip install datasets
```
If you use official api to acquire response, simply run
```
python <name of dataset>.py --<arguments>
```
### Arguments
```
--api_key
```
Your api key.
```
--result_dir
(optional)
```
The path of directory which stores evaluation results. By leaving this argument empty, the result will be stored in the same directory as the script.
```
--num_samples
```
The number of randomly selected samples from the dataset for testing.
```
--batch_size
```
Batch size. (will be 1 if this argument is left empty)
## Other way of implementation
If you are not using official deepseek api to acquire response or want to test other models, please adjust the function **deepseek_predict** in **predict.py**.
## Result
For each dataset, evaluation result will be stored as a JSONL file in the result directory.    
**overall_result.jsonl** in the result directory contains the f1 score and accuracy data for every dataset.
