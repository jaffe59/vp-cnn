import pandas
from statsmodels.sandbox.stats.runs import mcnemar

cnn_file = 'final_run.3.preds.txt'
lr_file = 'stats.1_prev_no_resp.csv'
dialogue_file = 'corrected.tsv'
def read_in_dialogues(dialogue_file):
    dialogue_indices = []
    dialogue_index = -1
    turn_index = -1
    records = []
    with open(dialogue_file) as l:
        for line in l:
            if line.startswith('#S'):
                dialogue_index += 1
                turn_index = 0
            else:
                dialogue_indices.append((dialogue_index, turn_index))
                records.append(line.strip())
                turn_index += 1
    return dialogue_indices, records

cnn_results = pandas.read_csv(cnn_file)

lr_results = pandas.read_csv(lr_file)

# x = cnn_results[(cnn_results['dial_id'] == lr_results['dial_id']) & (cnn_results['turn_id'] == lr_results['turn_id']
#                                                                      )& (cnn_results['correct'] != lr_results['correct'])]
k = 0
cnn_right_items = []
x, y = [], []
for index, cnn_item in cnn_results.iterrows():
    # print(cnn_item)
    lr_item = lr_results[(lr_results['dial_id']==cnn_item['dial_id']) & (lr_results['turn_id']==cnn_item['turn_id'])]
    # print(lr_item)
    # print(lr_item['correct'].iloc(0), cnn_item['correct'])
    # break
    if lr_item['correct'].iloc[0] != cnn_item['correct'] and not cnn_item['correct']:
        cnn_right_items.append((cnn_item.dial_id, cnn_item.turn_id))
    x.append(cnn_item.correct)
    y.append(lr_item['correct'].iloc[0])
# print(mcnemar(x,y))
indices, dialogues = read_in_dialogues(dialogue_file)
for item in cnn_right_items:
    print(dialogues[indices.index(item)])
    cnn_item = cnn_results[(lr_results['dial_id']==item[0]) & (lr_results['turn_id']==item[1])]
    print(cnn_item)
#
