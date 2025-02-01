import json

def read_jsonl(filename):
    data= []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def write_jsonl(file_path, lines, mode='a', encoding='utf-8'):
    with open(file_path, mode, encoding='utf-8') as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')