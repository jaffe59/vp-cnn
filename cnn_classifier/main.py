#! /usr/bin/env python
import datetime
import os
import model
import mydatasets
import numpy as np
import torchtext.data as data
import torchtext.datasets as datasets
import train
import vpdataset
from parse_args import parse_args
from chatscript_file_generator import *

# load VP dataset
def vp(text_field, label_field, foldid, num_experts=0, **kargs):
    # print('num_experts', num_experts)
    train_data, dev_data, test_data = vpdataset.VP.splits(text_field, label_field, foldid=foldid,
                                                          num_experts=num_experts)
    if num_experts > 0:
        text_field.build_vocab(train_data[0], dev_data[0], test_data, wv_type=kargs["wv_type"], wv_dim=kargs["wv_dim"],
                               wv_dir=kargs["wv_dir"], min_freq=kargs['min_freq'])
    else:
        text_field.build_vocab(train_data, dev_data, test_data, wv_type=kargs["wv_type"], wv_dim=kargs["wv_dim"],
                               wv_dir=kargs["wv_dir"], min_freq=kargs['min_freq'])
    # label_field.build_vocab(train_data, dev_data, test_data)
    kargs.pop('wv_type')
    kargs.pop('wv_dim')
    kargs.pop('wv_dir')
    kargs.pop("min_freq")
    # print(type(train_data), type(dev_data))
    if num_experts > 0:
        train_iter = []
        dev_iter = []
        for i in range(num_experts):
            this_train_iter, this_dev_iter, test_iter = data.Iterator.splits((train_data[i], dev_data[i], test_data),
                                                                             batch_sizes=(args.batch_size,
                                                                                          len(dev_data[i]),
                                                                                          len(test_data)), **kargs)
            train_iter.append(this_train_iter)
            dev_iter.append(this_dev_iter)
    else:
        train_iter, dev_iter, test_iter = data.Iterator.splits(
            (train_data, dev_data, test_data),
            batch_sizes=(args.batch_size,
                         len(dev_data),
                         len(test_data)),
            **kargs)
    return train_iter, dev_iter, test_iter


def char_tokenizer(mstring):
    return ['<pad>']*5 + list(mstring)


def check_vocab(field):
    itos = field.vocab.itos
    other_vocab = set()
    filename = '../sent-conv-torch/custom_word_mapping.txt'
    f = open(filename)
    for line in f:
        line = line.strip().split(" ")
        other_vocab.add(line[0])
    for word in itos:
        if word not in other_vocab:
            print(word)
    print('------')
    for word in other_vocab:
        if word not in itos:
            print(word)

def main():
    print("Beginning {0}-fold cross-validation...".format(args.xfolds))
    print("Logging the results in {}".format(args.log_file))
    log_file_handle = open(args.log_file, 'w')
    char_dev_fold_accuracies = []
    word_dev_fold_accuracies = []
    ensemble_dev_fold_accuracies = []
    char_test_fold_accuracies = []
    word_test_fold_accuracies = []
    ensemble_test_fold_accuracies = []
    orig_save_dir = args.save_dir
    update_args = True

    indices = calc_indices(args)
    labels = read_in_labels('data/labels.txt')
    dialogues = read_in_dialogues('data/wilkins_corrected.shuffled.51.indices')
    chats = read_in_chat('data/stats.1_prev_no_resp.csv', dialogues)

    for xfold in range(args.xfolds):
        print("Fold {0}".format(xfold))
        # load data
        print("\nLoading data...")

        tokenizer = data.Pipeline(vpdataset.clean_str)
        char_field = data.Field(lower=True, tokenize=char_tokenizer)
        word_field = data.Field(lower=True, tokenize=tokenizer)
        label_field = data.Field(sequential=False, use_vocab=False, preprocessing=int)

        train_iter, dev_iter, test_iter = vp(char_field, label_field, foldid=xfold, num_experts=args.num_experts,
                                             device=args.device, repeat=False, sort=False
                                             , wv_type=args.char_vector, wv_dim=args.char_embed_dim, wv_dir=args.char_emb_path, min_freq=1)
        train_iter_word, dev_iter_word, test_iter_word = vp(word_field, label_field, foldid=xfold,
                                                           num_experts=args.num_experts, device=args.device,
                                                           repeat=False, sort=False, wv_type=args.word_vector,
                                                           wv_dim=args.word_embed_dim, wv_dir=args.emb_path,
                                                           min_freq=args.min_freq)
        # check_vocab(word_field)
        # print(label_field.vocab.itos)


        args.class_num = 359
        args.cuda = args.yes_cuda and torch.cuda.is_available()  # ; del args.no_cuda
        if update_args == True:
            if isinstance(args.char_kernel_sizes,str):
                args.char_kernel_sizes = [int(k) for k in args.char_kernel_sizes.split(',')]
            args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), 'CHAR')
        else:
            args.save_dir = os.path.join(orig_save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), 'CHAR')

        print("\nParameters:", file=log_file_handle)
        for attr, value in sorted(args.__dict__.items()):
            print("\t{}={}".format(attr.upper(), value), file=log_file_handle)

        # char CNN training and dev
        args.embed_num = len(char_field.vocab)
        args.lr = args.char_lr
        args.l2 = args.char_l2
        args.epochs = args.char_epochs
        args.batch_size = args.char_batch_size
        args.dropout = args.char_dropout
        args.max_norm = args.char_max_norm
        args.kernel_num = args.char_kernel_num
        args.optimizer = args.char_optimizer

        print("\nParameters:")
        for attr, value in sorted(args.__dict__.items()):
            print("  {}={}".format(attr.upper(), value))

        if args.snapshot is None and args.num_experts == 0:
            char_cnn = model.CNN_Text(args, 'char', vectors=char_field.vocab.vectors)
        elif args.snapshot is None and args.num_experts > 0:
            char_cnn = [model.CNN_Text(args, 'char', vectors=char_field.vocab.vectors) for i in range(args.num_experts)]
        else:
            print('\nLoading model from [%s]...' % args.snapshot)
            try:
                char_cnn = torch.load(args.snapshot)
            except:
                print("Sorry, This snapshot doesn't exist.");
                exit()
        if args.num_experts > 0:
            acc, char_cnn = train.ensemble_train(train_iter, dev_iter, char_cnn, args,
                                                 log_file_handle=log_file_handle, always_norm=False)
        else:
            acc, char_cnn = train.train(train_iter, dev_iter, char_cnn, args, log_file_handle=log_file_handle)
        char_dev_fold_accuracies.append(acc)
        print("Completed fold {0}. Accuracy on Dev: {1} for CHAR".format(xfold, acc), file=log_file_handle)
        print("Completed fold {0}. Mean accuracy on Dev: {1} for CHAR".format(xfold, np.mean(acc)), file=log_file_handle)
        if args.eval_on_test:
            if args.num_experts > 0:
                result = train.ensemble_eval(test_iter, char_cnn, args, log_file_handle=log_file_handle)
            else:
                result = train.eval(test_iter, char_cnn, args, log_file_handle=log_file_handle)
            char_test_fold_accuracies.append(result)
            print("Completed fold {0}. Accuracy on Test: {1} for CHAR".format(xfold, result))
            print("Completed fold {0}. Accuracy on Test: {1} for CHAR".format(xfold, result), file=log_file_handle)


        log_file_handle.flush()

        # continue

        # Word CNN training and dev
        args.embed_num = len(word_field.vocab)
        args.lr = args.word_lr
        args.l2 = args.word_l2
        args.epochs = args.word_epochs
        args.batch_size = args.word_batch_size
        args.dropout = args.word_dropout
        args.max_norm = args.word_max_norm
        args.kernel_num = args.word_kernel_num
        args.optimizer = args.word_optimizer

        print("\nParameters:")
        for attr, value in sorted(args.__dict__.items()):
            print("  {}={}".format(attr.upper(), value))

        if update_args == True:
            # args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
            args.word_kernel_sizes = [int(k) for k in args.word_kernel_sizes.split(',')]
            args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), 'WORD')
        else:
            args.save_dir = os.path.join(orig_save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), 'WORD')

        if args.snapshot is None and args.num_experts == 0:
            word_cnn = model.CNN_Text(args, 'word', vectors=word_field.vocab.vectors)
        elif args.snapshot is None and args.num_experts > 0:
            word_cnn = [model.CNN_Text(args, 'word', vectors=word_field.vocab.vectors) for i in range(args.num_experts)]
        else:
            print('\nLoading model from [%s]...' % args.snapshot)
            try:
                word_cnn = torch.load(args.snapshot)
            except:
                print("Sorry, This snapshot doesn't exist.");
                exit()
        if args.num_experts > 0:
            acc, word_cnn = train.ensemble_train(train_iter_word, dev_iter_word, word_cnn, args,
                                                 log_file_handle=log_file_handle)
        else:
            acc, word_cnn = train.train(train_iter_word, dev_iter_word, word_cnn, args, log_file_handle=log_file_handle)
        word_dev_fold_accuracies.append(acc)
        print("Completed fold {0}. Accuracy on Dev: {1} for WORD".format(xfold, acc), file=log_file_handle)
        if args.eval_on_test:
            if args.num_experts > 0:
                result = train.ensemble_eval(test_iter_word, word_cnn, args, log_file_handle=log_file_handle)
            else:
                result = train.eval(test_iter_word, word_cnn, args, log_file_handle=log_file_handle)
            word_test_fold_accuracies.append(result)
            print("Completed fold {0}. Accuracy on Test: {1} for WORD".format(xfold, result))
            print("Completed fold {0}. Accuracy on Test: {1} for WORD".format(xfold, result), file=log_file_handle)

        # Ensemble training and dev
        if update_args == True:
            args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), 'LOGIT')
        else:
            args.save_dir = os.path.join(orig_save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), 'LOGIT')
        update_args = False
        #
        if args.snapshot is None:
            final_logit = model.StackingNet(args)
        else:
            print('\nLoading model from [%s]...' % args.snapshot)
            try:
                final_logit = torch.load(args.snapshot)
            except:
                print("Sorry, This snapshot doesn't exist.");
                exit()

        train_iter, dev_iter, test_iter = vp(char_field, label_field, foldid=xfold, device=args.device, repeat=False,
                                             shuffle=False, sort=False
                                             , wv_type=args.char_vector, wv_dim=args.char_embed_dim, wv_dir=args.char_emb_path, min_freq=1)
        train_iter_word, dev_iter_word, test_iter_word = vp(word_field, label_field, foldid=xfold,
                                                            device=args.device,
                                                            repeat=False, sort=False, shuffle=False,
                                                            wv_type=args.word_vector,
                                                            wv_dim=args.word_embed_dim, wv_dir=args.emb_path,
                                                            min_freq=args.min_freq)

        acc = train.train_final_ensemble(train_iter, dev_iter, train_iter_word, dev_iter_word, char_cnn, word_cnn, final_logit,
                                         args, log_file_handle=log_file_handle)
        ensemble_dev_fold_accuracies.append(acc)
        print("Completed fold {0}. Accuracy on Dev: {1} for LOGIT".format(xfold, acc), file=log_file_handle)
        if args.eval_on_test:
            result = train.eval_final_ensemble(test_iter, test_iter_word, char_cnn, word_cnn, final_logit, args,
                                               log_file_handle=log_file_handle, prediction_file_handle=prediction_file_handle,
                                               labels=labels, chats=chats, dialogues=dialogues, indices=indices, fold_id=xfold)
            ensemble_test_fold_accuracies.append(result)

            print("Completed fold {0}. Accuracy on Test: {1} for LOGIT".format(xfold, result))
            print("Completed fold {0}. Accuracy on Test: {1} for LOGIT".format(xfold, result), file=log_file_handle)

        log_file_handle.flush()

    print("CHAR mean accuracy is {}, std is {}".format(np.mean(char_dev_fold_accuracies), np.std(char_dev_fold_accuracies)))
    print("WORD mean accuracy is {}, std is {}".format(np.mean(word_dev_fold_accuracies), np.std(word_dev_fold_accuracies)))
    print("LOGIT mean accuracy is {}, std is {}".format(np.mean(ensemble_dev_fold_accuracies), np.std(ensemble_dev_fold_accuracies)))
    print("CHAR mean accuracy is {}, std is {}".format(np.mean(char_dev_fold_accuracies), np.std(char_dev_fold_accuracies)), file=log_file_handle)
    print("WORD mean accuracy is {}, std is {}".format(np.mean(word_dev_fold_accuracies), np.std(word_dev_fold_accuracies)),
         file=log_file_handle)
    print("LOGIT mean accuracy is {}, std is {}".format(np.mean(ensemble_dev_fold_accuracies), np.std(ensemble_dev_fold_accuracies)), file=log_file_handle)

    if char_test_fold_accuracies or word_test_fold_accuracies:
        print("CHAR mean accuracy is {}, std is {}".format(np.mean(char_test_fold_accuracies), np.std(char_test_fold_accuracies)))
        print("WORD mean accuracy is {}, std is {}".format(np.mean(word_test_fold_accuracies),
                                                          np.std(word_test_fold_accuracies)))
        print("LOGIT mean accuracy is {}, std is {}".format(np.mean(ensemble_test_fold_accuracies), np.std(ensemble_test_fold_accuracies)))

        print("CHAR mean accuracy is {}, std is {}".format(np.mean(char_test_fold_accuracies), np.std(char_test_fold_accuracies)), file=log_file_handle)
        print("WORD mean accuracy is {}, std is {}".format(np.mean(word_test_fold_accuracies),
                                                          np.std(word_test_fold_accuracies)), file=log_file_handle)
        print("LOGIT mean accuracy is {}, std is {}".format(np.mean(ensemble_test_fold_accuracies), np.std(ensemble_test_fold_accuracies)), file=log_file_handle)

    log_file_handle.close()
    prediction_file_handle.close()

if __name__ == '__main__':
    args = parse_args()

    prediction_file_handle = open(args.prediction_file_handle, 'w')
    print(
        'dial_id,turn_id,predicted,correct,prob,entropy,confidence,ave_prob,ave_logporb,chatscript_prob,chatscript_rank',
        file=prediction_file_handle)

    main()