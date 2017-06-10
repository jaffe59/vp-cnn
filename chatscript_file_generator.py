import math
import torch
import scipy.stats as stats
def calc_indices(args):
    #calc fold indices
    indices = []
    numfolds = args.xfolds
    len_dataset = 4330
    fold_size = math.ceil(len_dataset/numfolds)
    for fold in range(numfolds):
        startidx = fold*fold_size
        endidx = startidx+fold_size if startidx+fold_size < len_dataset else len_dataset
        indices.append((startidx, endidx))
    return indices

def read_in_labels(labels_file):
    labels = []
    with open(labels_file) as l:
        for line in l:
            line = line.strip().split("\t")
            print(line)
            labels.append('_'.join(line[1]))
    return labels

def read_in_dialogues(dialogue_file):
    dialogue_indices = []
    dialogue_index = -1
    turn_index = -1
    with open(dialogue_file) as l:
        for line in l:
            if line.startswith('#S'):
                dialogue_index += 1
                turn_index = 0
            else:
                turn_index += 1
                dialogue_indices.append((dialogue_index, turn_index))
    return dialogue_indices

def read_in_chat(chat_file):
    chats = []
    with open(chat_file) as c:
        for line in c:
            if line.startswith('dia'):
                continue
            else:
                line = line.strip().split(',')
                chats.append((line[-2], line[-1]))
    return chats

def print_test_features(tensor, confidence, target, dialogue_indices, labels, indices, fold_id, chats, feature_file):
    # dial_id, turn_id, predicted_label, correct_bool, prob, entropy, confidence, chat_prob, chat_rank
    tensor = torch.exp(tensor)
    probs, predicted = torch.max(tensor, 1)
    predicted = predicted.view(target.size()).data
    probs = probs.view(target.size()).data
    corrects = predicted == target.data
    confidence = confidence.squeeze().data.cpu().numpy()
    tensor = tensor.squeeze().data.cpu().numpy()
    start_id, end_id = indices[fold_id]
    for ind, val in enumerate(corrects):
        item = []
        item_id = start_id+ind
        dialogue_index, turn_index = dialogue_indices[item_id]
        item += [dialogue_indices, turn_index]
        item.append(labels[predicted[ind]])
        item.append(str(val))
        item.append(probs[ind])
        item.append(stats.entropy(tensor[ind]))
        item.append(confidence[ind, predicted[ind]])
        item.append(chats[item_id][0])
        item.append(chats[item_id][1])
        print(','.join(item), file=feature_file)





