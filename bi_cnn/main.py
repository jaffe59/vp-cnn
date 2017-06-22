from bicnn_model import bi_CNN_Text
import torch
from cnn_classifier.parse_args import parse_args
from torch.autograd import Variable
from torch import nn
from cnn_classifier.main import char_tokenizer
from torchtext import data
from bicnn_vpdataset import *
from cnn_classifier.vpdataset import clean_str
import train

args = parse_args()
prediction_file_handle = open(args.prediction_file_handle, 'w')
log_file_handle = open(args.log_file, 'w')

print("\nLoading data...")

tokenizer = data.Pipeline(clean_str)
char_field = data.Field(lower=True, tokenize=char_tokenizer)
word_field = data.Field(lower=True, tokenize=tokenizer)
label_field = data.Field(sequential=False, use_vocab=False, preprocessing=int)

train_iters, dev_iters, test_iters = vp_bicnn(char_field, label_field, args=args,
                                           num_experts=args.num_experts,
                                           device=args.device, repeat=False, sort=False, shuffle=False,
                                           wv_type=args.char_vector, wv_dim=args.char_embed_dim,
                                           wv_dir=args.char_emb_path,
                                           min_freq=1)
# train_iter_word, dev_iter_word, test_iter_word = vp_bicnn(word_field, label_field, args=args,
#                                                           num_experts=args.num_experts, device=args.device,
#                                                           repeat=False, sort=False, wv_type=args.word_vector,
#                                                           wv_dim=args.word_embed_dim, wv_dir=args.emb_path,
#                                                           min_freq=args.min_freq)
print(train_iters)
for xfold in range(args.xfolds):
    if xfold != 9:
        continue
    train_iter = train_iters[xfold]
    dev_iter = dev_iters[xfold]
    test_iter = test_iters[xfold]
    args.cuda = args.yes_cuda and torch.cuda.is_available()
    model = bi_CNN_Text(args, 'char', vectors=None)
    train.train(train_iter, dev_iter, model, args)


    # print("\nParameters:", file=log_file_handle)
    # for attr, value in sorted(args.__dict__.items()):
    #     print("\t{}={}".format(attr.upper(), value), file=log_file_handle)
    #
