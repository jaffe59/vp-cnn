
label_filename = '../data/labels.txt'
question_filename = '../data/wilkins_corrected.shuffled.51.txt'
output_filename = '../data/wilkins_corrected.shuffled.51.bicnn.txt'

labels = {}
with open(label_filename) as l:
    for line in l:
        index, string = line.strip().split('\t')
        labels[index] = string


with open(question_filename) as q, open(output_filename, 'w') as w:
    for line in q:
        index, string = line.strip().split('\t')
        for label_index in labels.keys():
            if label_index == index:
                y = '1'
            else:
                y = '-1'
            print('\t'.join([y, string, labels[label_index]]), file=w)


