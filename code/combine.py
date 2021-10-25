import argparse
import os
import json
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--project_name', type=str)
parser.add_argument('--run_name', type=str)
args = parser.parse_args()

path = f'../predict/{args.project_name}/{args.run_name}'
all_data = []
for k in range(1, 6):
    unit_path = os.path.join(path, str(k), f'nbest_predictions_{args.run_name}.json')
    with open(unit_path, 'r') as f:
        data = json.load(f)
    all_data.append(data)

result = {}
ids = list(all_data[0].keys())
for id in ids:
    pred = defaultdict(int)
    for data in all_data:
        temp = defaultdict(int)
        for d in data[id]:
            sl, el, text, p = d.values()
            temp[text] = max(temp[text], p)
        for k, v in temp.items():
            pred[k] += v
    answer = sorted(pred.items(), key = lambda x: -x[1])[0][0]
    result[id] = answer

with open(f'../predict/{args.project_name}/{args.run_name}/combined_predictions.json', 'w') as f:
    json.dump(result, f)