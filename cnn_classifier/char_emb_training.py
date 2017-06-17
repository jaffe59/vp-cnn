from gensim.models import Word2Vec
import sys
import gzip
import numpy as np
import pickle

def characterfy_corpus(corpus):
    if sent_file.endswith('gz'):
        f = gzip.open(sent_file, "rt")
    else:
        f = open(sent_file)
    data = []
    for line in f:
        line = line.strip()
        data.append(list(line))
    return data

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
            print(word+'\t'+arr, file=o)


if __name__ == '__main__':
    sent_file = sys.argv[1]
    char_embed_file = sys.argv[2]
    char_embeddings = create_embeddings(sent_file)
    printout_embeds(char_embeddings, char_embed_file)