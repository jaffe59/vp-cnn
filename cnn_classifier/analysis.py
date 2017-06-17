import sys
import matplotlib.pyplot as plt
import numpy as np
import re
import scipy.stats
from matplotlib import cm

log_file = sys.argv[1]


# quintile	chatscript	LR	word cnn	ensembled cnn	shuffled ensembled cnn
# least frequent	69.9%	41.7%	31.00%	42%	46.50%
# 79.0%	72.8%	71.00%	73.60%	74.61%
# 79.3%	88.7%	88.50%	89.50%	89.53%
# 86.5%	93.4%	92.90%	93.80%	93.83%
# most frequent	84.5%	90.2%	90.80%	91.50%	91.47%

chat_script = [69.9, 79.0, 79.3, 86.5, 84.5]
LR = [41.7, 72.8, 88.7, 93.4, 90.2]
# word_cnn = [31.0,71.0, 88.5,92.9,90.8]
ensembled = [46.5,74.61,89.53,93.83,91.47]

fig, ax = plt.subplots()
colors = [cm.tab20c(x) for x in np.linspace(0.0, 1.0, 3)]
# rect2 = ax.bar(ind+width*1, char, width, color=colors[1])
# rect4 = ax.bar(ind+width*3, word, width, color=colors[3])
# # rect3 = ax.bar(ind+2*width, logit, width, color='g')
# rect1 = ax.bar(ind, single_char, width, color=colors[0])
# rect3 = ax.bar(ind+width*2, single_word, width, color=colors[2])
#
# ax.set_ylabel('Accuracy %')
# ax.set_title("Accuracy of different folds for submodels and ensembles")
# ax.set_xticks(ind+width*2)
# ax.set_ylim(70, 83)
# ax.set_xticklabels(('fold '+str(x) for x in range(10)))
# ax.legend((rect1[0], rect2[0], rect3[0],rect4[0]), ('BestCharCNN',   'CharCNNEns','BestWordCNN','WordCNNEns'),loc=1)
ind = np.arange(5)
width = 0.30
rect1 = ax.bar(ind, chat_script, width, color=colors[1])
rect2 = ax.bar(ind + width * 1, LR, width, color=colors[2])
rect3 = ax.bar(ind + 2 * width, ensembled, width, color=colors[0])
# rect1 = ax.bar(ind, single_char, width, color='magenta')
# rect3 = ax.bar(ind+width*2, single_word, width, color='green')

ax.set_ylabel('Accuracy %')
ax.set_title("Average accuracy of different systems in groups of label frequency quintiles")
ax.set_xticks(ind + width * 1)
ax.set_ylim(40, 100)
ax.set_xlabel('Label frequency quintile')
ax.set_xticklabels(('Least frequent', '', '', '', 'Most frequent'))
ax.legend((rect1[0], rect2[0], rect3[0]), ('ChatScript', 'Baseline', 'Stacked'), loc=1)

plt.show()


single_word_dev = [[77.1208,80.7198,77.1208,77.3779,78.9203],[78.4062,79.6915,78.6632,77.635,78.4062],[79.9486,76.8638,79.6915,77.3779,78.6632],[79.1774,77.1208,81.7481,78.4062,79.6915],[78.9203,76.3496,80.2057,77.1208,79.1774],[79.9486,76.8638,81.7481,77.635,78.9203],[77.635,78.4062,79.6915,78.1491,77.3779],[78.6632,76.6067,80.4627,76.0925,78.9203],[79.1774,75.8355,79.9486,77.3779,77.1208],[77.635,76.6067,79.9486,77.6350,78.6632]]
single_char_dev = [[76.6067,79.4344,77.635,76.6067,78.5219],[77.3779,78.9203,78.9203,77.892,75.8355],[79.4344,78.9203,78.1491,77.3779,76.6067],[78.1491,77.1208,79.1774,78.4062,75.5784],[76.8638,77.635,78.1491,76.8638,76.3496],[78.6632,78.6632,79.6915,77.1208,75.3213],[78.6632,77.8920,78.6632,76.8638,77.892],[76.6067,77.892,79.1774,77.635,77.892],[78.4062,76.8638,77.8920,78.6632,76.8638],[76.8638,75.3213,80.2057,77.1208,78.6632]]

with open(log_file) as l:
    word = []
    char = []
    logit = []
    k = 0
    single_word = []

    single_char = []
    temp = []
    for line in l:
        if line.startswith(' ') or line.startswith('Para') or line.startswith('Batch'):
            continue
        elif line.startswith('Evaluation model'):
            temp.append(float(re.search("([0-9]{2}\.[0-9]{4})\%", line).group(1)))
            if len(temp) == 5:
                if k == 0:
                    dev_set = single_char_dev[len(single_char)]
                    best_index = np.argmax(dev_set)
                    single_char.append(temp[best_index])
                    k = 1
                else:
                    dev_set = single_word_dev[len(single_word)]
                    best_index = np.argmax(dev_set)
                    single_word.append(temp[best_index])
                    k = 0
                temp = []
        else:
            if line.startswith('Complete'):
                line = line.strip().split(' ')
                if line[-1] == 'CHAR':
                    char.append(float(line[6]))
                elif line[-1] == 'WORD':
                    word.append(float(line[6]))
                elif line[-1] == 'LOGIT':
                    logit.append(float(line[6]))
                else:
                    raise Exception
    ind = np.arange(10)
    width = 0.15

    print(np.mean(single_char), np.mean(char), np.mean(single_word), np.mean(word))

    print(scipy.stats.ttest_ind(single_word, logit))
    print(scipy.stats.ttest_ind(single_char, logit))
    #
    # print(scipy.stats.ttest_ind(word, logit))
    # print(scipy.stats.ttest_ind(char, logit))
    #
    # print(scipy.stats.ttest_1samp(single_word, sum(word)/len(word)))
    # print(scipy.stats.ttest_1samp(single_char, sum(char)/len(char)))
    #
    # print(scipy.stats.ttest_1samp(word, sum(logit)/len(logit)))
    # print(scipy.stats.ttest_1samp(char, sum(logit)/len(logit)))

    fig, ax = plt.subplots()
    colors = [cm.Accent(x) for x in np.linspace(0.0, 1.0, 5)]
    rect2 = ax.bar(ind+width*1, char, width, color=colors[1])
    rect4 = ax.bar(ind+width*3, word, width, color=colors[3])
    # rect3 = ax.bar(ind+2*width, logit, width, color='g')
    rect1 = ax.bar(ind, single_char, width, color=colors[0])
    rect3 = ax.bar(ind+width*2, single_word, width, color=colors[2])
    rect5 = ax.bar(ind+4*width, logit, width, color=colors[4])

    ax.set_ylabel('Accuracy %')
    ax.set_title("Accuracy of different folds for submodels and ensembles")
    ax.set_xticks(ind+width*2)
    ax.set_ylim(70, 83)
    ax.set_xticklabels(('fold '+str(x) for x in range(10)))
    ax.legend((rect1[0], rect2[0], rect3[0],rect4[0], rect5[0]), ('BestCharCNN',   'CharCNNEns','BestWordCNN','WordCNNEns', 'Stacked'),loc=1)

    # rect1 = ax.bar(ind, char, width, color=colors[1])
    # rect2 = ax.bar(ind+width*1, word, width, color=colors[3])
    # rect3 = ax.bar(ind+2*width, logit, width, color=colors[0])
    # # rect1 = ax.bar(ind, single_char, width, color='magenta')
    # # rect3 = ax.bar(ind+width*2, single_word, width, color='green')
    #
    # ax.set_ylabel('Accuracy %')
    # ax.set_title("Accuracy of different folds for ensembles and the stacked model")
    # ax.set_xticks(ind+width*1)
    # ax.set_ylim(70, 83)
    # ax.set_xticklabels(('fold '+str(x) for x in range(10)))
    # ax.legend((rect1[0], rect2[0], rect3[0]), (  'CharCNNEns','WordCNNEns' ,'Stacked'),loc=1)
    #
    #
    plt.show()



