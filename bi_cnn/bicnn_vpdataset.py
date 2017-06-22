from torchtext import data
import os
import datetime
import math, random


def vp_bicnn(text_field, label_field, args, num_experts=0, **kargs):
    print(text_field, label_field, args.xfolds, num_experts)
    xfolds_data = VP_BICNN.splits(text_field, label_field, num_folds=args.xfolds,
                                                      num_experts=num_experts)
    if num_experts > 0:
        text_field.build_vocab(xfolds_data[0][0][0], xfolds_data[0][1][0], xfolds_data[0][2], wv_type=kargs["wv_type"], wv_dim=kargs["wv_dim"],
                               wv_dir=kargs["wv_dir"], min_freq=kargs['min_freq'])
    else:
        text_field.build_vocab(xfolds_data[0][0], xfolds_data[0][1], xfolds_data[0][2], wv_type=kargs["wv_type"], wv_dim=kargs["wv_dim"],
                               wv_dir=kargs["wv_dir"], min_freq=kargs['min_freq'])
    # label_field.build_vocab(train_data, dev_data, test_data)
    kargs.pop('wv_type')
    kargs.pop('wv_dim')
    kargs.pop('wv_dir')
    kargs.pop("min_freq")
    # print(type(train_data), type(dev_data))
    if num_experts > 0:
        train_iters = []
        dev_iters = []
        test_iters = []
        for fold in args.xfolds:
            for i in range(num_experts):
                this_train_iter, this_dev_iter, test_iter = data.Iterator.splits((xfolds_data[fold][0][i], xfolds_data[fold][1][i], xfolds_data[fold][2]),
                                                                                 batch_sizes=(args.batch_size,
                                                                                              len(xfolds_data[fold][1][i]),
                                                                                              len(xfolds_data[fold][2])), **kargs)
                train_iters.append(this_train_iter)
                dev_iters.append(this_dev_iter)
            test_iters.append(test_iter)
    else:
        train_iters = []
        dev_iters = []
        test_iters = []
        for fold in range(args.xfolds):
            train_iter, dev_iter, test_iter = data.Iterator.splits(
                (xfolds_data[fold][0], xfolds_data[fold][1], xfolds_data[fold][2]),
                batch_sizes=(args.batch_size,
                             len(xfolds_data[fold][1]),
                             len(xfolds_data[fold][2])),
                **kargs)
            train_iters.append(train_iter)
            dev_iters.append(dev_iter)
            test_iters.append(test_iter)
    return train_iters, dev_iters, test_iters


class VP_BICNN(data.Dataset):
    """modeled after Shawn1993 github user's Pytorch implementation of Kim2014 - cnn for text categorization"""

    filename = "../data/wilkins_corrected.shuffled.51.bicnn.txt"

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
        # no preprocessing needed
        fields = [('s1', text_field), ('s2', text_field), ('label', label_field)]

        if examples is None:
            path = self.dirname if path is None else path
            examples = []
            with open(os.path.join(path, self.filename)) as f:
                lines = f.readlines()
                # pdb.set_trace()
                k = 0
                for line in lines:
                    k += 1
                    if k % (359*10) == 0:
                        print('{}: processed {} examples'.format(datetime.datetime.now().strftime('%H:%M:%S'), k))
                        break
                    label, s1, s2 = line.split("\t")
                    this_example = data.Example.fromlist([s1, s2, label], fields)
                    examples += [this_example]

                    # assume "target \t source", one instance per line
        # print(examples[0].text)
        super(VP_BICNN, self).__init__(examples, fields, **kwargs)

    @classmethod
    # def splits(cls, text_field, label_field, dev_ratio=.1, shuffle=True ,root='.', **kwargs):
    def splits(cls, text_field, label_field, num_folds=10, dev_ratio=.1, shuffle=False, root='.',
               num_experts=0, **kwargs):

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
        # path = cls.download_or_unzip(root)
        # examples = cls(text_field, label_field, path=path, **kwargs).examples
        examples = cls(text_field, label_field, path=root, **kwargs).examples
        if shuffle: random.shuffle(examples)
        fields = [('s1', text_field), ('s2', text_field), ('label', label_field)]
        label_examples = []
        label_filename = '../data/labels.txt'
        labels = {}
        with open(label_filename) as f:
            lines = f.readlines()
            # pdb.set_trace()
            for line in lines:
                label, text = line.split("\t")
                labels[label] = text
            for label in labels.keys():
                for label2 in labels.keys():
                    this_example = data.Example.fromlist(
                        [labels[label], labels[label2], '1' if label == label2 else '-1'], fields)
                    label_examples += [this_example]
        if num_folds <= 1:
            dev_index = -1 * int(dev_ratio * len(examples))
            return (cls(text_field, label_field, examples=examples[:dev_index]),
                    cls(text_field, label_field, examples=examples[dev_index:])
                    )
        else:
            # get all folds
            fold_size = math.ceil(len(examples) / num_folds)
            folds = []
            for fold in range(num_folds):
                startidx = fold * fold_size
                endidx = startidx + fold_size if startidx + fold_size < len(examples) else len(examples)
                folds += [examples[startidx:endidx]]
            fold_datasets = []
            for foldid in range(num_folds):

                test = folds[foldid]
                # print(len(traindev[:dev_index]), 'num_experts', num_experts)
                if num_experts > 0:
                    assert num_experts <= 5
                    trains = []
                    devs = []
                    # print(dev_length)
                    for i in range(num_experts):
                        while True:
                            dev_id = random.randint(0, num_folds-1)
                            if dev_id != foldid:
                                break
                        train = [fold for idx, fold in enumerate(folds) if (idx != foldid and idx != dev_id)]
                        train = [item for sublist in train for item in sublist]
                        dev = folds[dev_id]
                        devs.append(
                            cls(text_field, label_field, examples=dev))
                        trains.append(cls(text_field, label_field, examples=train + label_examples))
                    fold_datasets.append((trains, devs, cls(text_field, label_field, examples=test)))
                else:
                    while True:
                        dev_id = random.randint(0, num_folds-1)
                        if dev_id != foldid:
                            break
                    train = [fold for idx, fold in enumerate(folds) if idx != foldid and idx != dev_id]
                    train = [item for sublist in train for item in sublist]
                    print(dev_id)
                    dev = folds[dev_id]
                    fold_datasets.append((cls(text_field, label_field, examples=train + label_examples),
                                          cls(text_field, label_field, examples=dev),
                                          cls(text_field, label_field, examples=test)))
            return fold_datasets
