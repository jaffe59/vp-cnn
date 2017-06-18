from gensim.models import Word2Vec
import gzip
import numpy as np
import pickle
import argparse
import os
'''
this standalone script is for generating character-based embeddings trained on custom data.
it uses gensim to train with w2v skipgram
you need to pass in three arguments to this file:
1. the path to the sentences file
2. the name of the output embedding txt file
3. the size of the char embedding [default is 16]
'''
class characterfied_corpus:
    def __init__(self, sent_file):
        self.file_name = sent_file
        if sent_file.endswith('gz'):
            self.f = gzip.open(sent_file, "rt")
        else:
            self.f = open(sent_file)

    def __iter__(self):
        while True:
            line = self.f.readline()
            if not line:
                self.f.seek(0)
                raise StopIteration
            line = list(line.strip())
            yield line

def create_embeddings(data, size=16):
    char_embs_binary = 'char_embs_'+str(size)+'.pkl'
    model = Word2Vec(data, size, workers=15, sg=1)
    vectors = model.wv
    pickle.dump(vectors, open(char_embs_binary, 'wb'))
    return vectors

def printout_embeds(char_embs, char_embs_file):
    with open(char_embs_file, 'w') as o:
        for word in char_embs.vocab:
            arr = np.array_str(char_embs[word])
            arr = arr[1:-1]
            print(word+' '+arr, file=o)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Character embedding trainer')
    parser.add_argument('-sent-file', type=str, default=os.getcwd(), help='path of the sentences file [default: .]')
    parser.add_argument('-char-emb-file', type=str, default='emb.txt', help='name of the output char embedding file [default: emb.txt]')
    parser.add_argument('-char-emb-size', type=int, default=16, help='size of the char embedding [default:16]')
    args = parser.parse_args()
    sent_file = args.sent_file
    char_embed_file = args.char_emb_file
    emb_size = args.char_emb_size

    data = characterfied_corpus(sent_file)
    char_embeddings = create_embeddings(data, size=emb_size)
    printout_embeds(char_embeddings, char_embed_file)