from chatscript_file_generator import *
import random

def shuffle_data(dataset_list, dialogues, shuffled_data_file, indices_file):
    len_dataset = len(dataset_list)
    indices = list(range(len_dataset))
    random.shuffle(indices)
    shuffled_data_file_handle = open(shuffled_data_file, 'w')
    indices_file_handle = open(indices_file, 'w')
    for index in indices:
        print(dataset_list[index].strip(), file=shuffled_data_file_handle)
        print(dialogues[index], file=indices_file_handle)
    indices_file_handle.close()
    indices_file_handle.close()

def main(data_file, dialogue_file, shuffled_data_file, indices_file):
    dialogues = read_in_dialogues(dialogue_file)
    data_list = open(data_file).readlines()
    shuffle_data(data_list, dialogues, shuffled_data_file, indices_file)

if __name__ == '__main__':
    dialogue_file = 'corrected.tsv'
    data_file = 'wilkins_corrected.tsv'
    a = random.randint(0, 100)
    shuffled_data_file = 'wilkins_corrected.shuffled.'+str(a)+'.txt'
    indices_file = 'wilkins_corrected.shuffled.'+str(a)+'.indices'
    main(data_file, dialogue_file,shuffled_data_file, indices_file)