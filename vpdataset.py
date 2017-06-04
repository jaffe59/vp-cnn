from torchtext import data
import os
import pdb
import random
import math
import re

class VP(data.Dataset):
    """modeled after Shawn1993 github user's Pytorch implementation of Kim2014 - cnn for text categorization"""

    filename = "wilkins_corrected.shuffled.tsv"

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, text_field, label_field, path=None, examples=None, **kwargs):
        """Create a virtual patient (VP) dataset instance given a path and fields.

        Arguments:
            text_field: The field that will be used for text data.
            label_field: The field that will be used for label data.
            path: Path to the data file.
            examples: The examples contain all the data.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        #no preprocessing needed 
        fields = [('text', text_field), ('label', label_field)]

        if examples is None:
                path = self.dirname if path is None else path
                examples = []
                with open(os.path.join(path, self.filename)) as f:
                    lines = f.readlines()
                    #pdb.set_trace()
                    examples += [
                        data.Example.fromlist([clean_str(line.split("\t")[1]), line.split("\t")[0]], fields) for line in lines] #in f]
                    #assume "target \t source", one instance per line
        # print(examples[0].text)
        super(VP, self).__init__(examples, fields, **kwargs)
        

    @classmethod
    #def splits(cls, text_field, label_field, dev_ratio=.1, shuffle=True ,root='.', **kwargs):
    def splits(cls, text_field, label_field, numfolds=10, foldid=None, dev_ratio=.1, shuffle=False ,root='.', **kwargs):
        
        """Create dataset objects for splits of the VP dataset.

        Arguments:
            text_field: The field that will be used for the sentence.
            label_field: The field that will be used for label data.
            dev_ratio: The ratio that will be used to get split validation dataset.
            shuffle: Whether to shuffle the data before split.
            root: The root directory that the dataset's zip archive will be
                expanded into; therefore the directory in whose trees
                subdirectory the data files will be stored.
            train: The filename of the train data. Default: 'train.txt'.
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """
        #path = cls.download_or_unzip(root)
        #examples = cls(text_field, label_field, path=path, **kwargs).examples
        examples = cls(text_field, label_field, path=root, **kwargs).examples
        if shuffle: random.shuffle(examples)
        
        if foldid==None:
            dev_index = -1 * int(dev_ratio*len(examples))
            return (cls(text_field, label_field, examples=examples[:dev_index]),
                    cls(text_field, label_field, examples=examples[dev_index:]))
        else:
            #get all folds
            fold_size = math.floor(len(examples)/numfolds)
            folds = []
            for fold in range(numfolds):
                startidx = fold*fold_size
                endidx = startidx+fold_size
                folds += [examples[startidx:endidx]]

            #take all folds except foldid as training/dev
            traindev = [fold for idx, fold in enumerate(folds) if idx != foldid]
            traindev = [item for sublist in traindev for item in sublist]
            dev_index = -1 * int(dev_ratio*len(traindev))

            #test will be entire held out section (foldid)
            test = folds[foldid]

            return (cls(text_field, label_field, examples=traindev[:dev_index]),
                    cls(text_field, label_field, examples=traindev[dev_index:]),
                    cls(text_field, label_field, examples=test))

def clean_str(string):
  """
  Tokenization/string cleaning for all datasets except for SST.
  """
  string = re.sub("[^A-Za-z0-9(),!?\'\`]", " ", string)
  string = re.sub("\'s", " \'s", string)
  string = re.sub("\'m", " \'m", string)
  string = re.sub("\'ve", " \'ve", string)
  string = re.sub("n\'t", " n\'t", string)
  string = re.sub("\'re", " \'re", string)
  string = re.sub("\'d", " \'d", string)
  string = re.sub("\'ll", " \'ll", string)
  string = re.sub(",", " , ", string)
  string = re.sub("!", " ! ", string)
  string = re.sub("\(", " ( ", string)
  string = re.sub("\)", " ) ", string)
  string = re.sub("\?", " ? ", string)
  string = re.sub("\s{2,}", " ", string)
  return string.strip().lower()