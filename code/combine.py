import argparse
import os
import json
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--project_name', type=str)
parser.add_argument('--run_name', type=str)
args = parser.parse_args()

'''
combine.py
fold를 구현하기 위해 내부적으로 변경하는 것은 소모적인 일이라 판단되어,
fold를 독립적으로 수행한 뒤 얻어지는 nbest_prediction.json 파일을 가지고
soft voting 하는 방식으로 kfold의 최종 result를 얻습니다.
'''

# 5개의 파일 로드
path = f'../predict/{args.project_name}/{args.run_name}'
all_data = []
for k in range(1, 6):
    unit_path = os.path.join(path, str(k), f'nbest_predictions_{args.run_name}.json')
    with open(unit_path, 'r') as f:
        data = json.load(f)
    all_data.append(data)

# 한 개의 폴드에서, 동일한 단어에 대해서는 제일 높은 확률로 해당 단어의 확률을 사용
# 다섯 개의 폴드에서 모든 확률을 합한 뒤 최고 확률의 단어 저장
result = {}
score = {}
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
    answer = sorted(pred.items(), key=lambda x: -x[1])[0][0]
    result[id] = answer
    score[id] = sorted(pred.items(), key=lambda x: -x[1])

# 결과 파일 저장
with open(f'../predict/{args.project_name}/{args.run_name}/combined_predictions.json', 'w') as f:
    json.dump(result, f)

# 결과에 
with open(f'../predict/{args.project_name}/{args.run_name}/combined_score.json', 'w') as f:
    json.dump(score, f)
