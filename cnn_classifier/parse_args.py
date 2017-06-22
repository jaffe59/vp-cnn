import argparse
import datetime
import os

def parse_args():
    parser = argparse.ArgumentParser(description='CNN text classificer')
    # learning
    parser.add_argument('-lr', type=float, default=1.0, help='initial learning rate [default: 1.0]') # 1e-3
    parser.add_argument('-word-lr', type=float, default=1.0, help='initial learning rate [default: 1.0]') # 1e-3
    parser.add_argument('-char-lr', type=float, default=1.0, help='initial learning rate [default: 1.0]') # 1e-3
    parser.add_argument('-l2', type=float, default=0.0, help='l2 regularization strength [default: 0.0]') # 1e-6
    parser.add_argument('-word-l2', type=float, default=0.0, help='l2 regularization strength [default: 0.0]') # 1e-6
    parser.add_argument('-char-l2', type=float, default=0.0, help='l2 regularization strength [default: 0.0]') # 1e-6
    parser.add_argument('-epochs', type=int, default=25, help='number of epochs for train [default: 25]')
    parser.add_argument('-word-epochs', type=int, default=25, help='number of epochs for train [default: 25]')
    parser.add_argument('-char-epochs', type=int, default=25, help='number of epochs for train [default: 25]')
    parser.add_argument('-batch-size', type=int, default=50, help='batch size for training [default: 50]')
    parser.add_argument('-word-batch-size', type=int, default=50, help='batch size for training [default: 50]')
    parser.add_argument('-char-batch-size', type=int, default=50, help='batch size for training [default: 50]')
    parser.add_argument('-log-interval', type=int, default=1,
                        help='how many steps to wait before logging training status [default: 1]')
    parser.add_argument('-log-file', type=str, default=datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + 'result.txt',
                        help='the name of the file to store results')
    parser.add_argument('-verbose', action='store_true', default=False, help='logging verbose info of training process')
    # parser.add_argument('-verbose-interval', type=int, default=5000, help='steps between two verbose logging')
    parser.add_argument('-test-interval', type=int, default=500,
                        help='how many steps to wait before testing [default: 100]')
    parser.add_argument('-eval-on-test', action='store_true', default=False, help='run evaluation on test data?')
    parser.add_argument('-save-interval', type=int, default=5000, help='how many steps to wait before saving [default:500]')
    parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
    # data
    parser.add_argument('-shuffle', action='store_true', default=True, help='shuffle the data every epoch')
    # model
    parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
    parser.add_argument('-char-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
    parser.add_argument('-word-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
    parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]') # 0.0
    parser.add_argument('-word-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]') # 0.0
    parser.add_argument('-char-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]') # 0.0
    parser.add_argument('-char-embed-dim', type=int, default=16, help='number of char embedding dimension [default: 128]')
    parser.add_argument('-word-embed-dim', type=int, default=300, help='number of word embedding dimension [default: 300]')

    parser.add_argument('-kernel-num', type=int, default=100, help='number of each kind of kernel')
    parser.add_argument('-word-kernel-num', type=int, default=300, help='number of each kind of kernel')
    parser.add_argument('-char-kernel-num', type=int, default=400, help='number of each kind of kernel')
    # parser.add_argument('-kernel-sizes', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')
    # parser.add_argument('-char-kernel-sizes', type=str, default='2,3,4,5,6', help='comma-separated kernel size to use for char convolution')
    parser.add_argument('-char-kernel-sizes', metavar='N', type=int, nargs='+', default=[2,3,4,5,6], help='comma-separated kernel size to use for char convolution')
    parser.add_argument('-word-kernel-sizes', metavar='N', type=int, nargs='+', default=[3,4,5], help='comma-separated kernel size to use for word convolution')

    # parser.add_argument('-word-kernel-sizes', type=str, default='3,4,5', help='comma-separated kernel size to use for word convolution')
    parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')

    # device
    parser.add_argument('-device', type=int, default=0, help='device to use for iterate data, -1 mean cpu [default: 0]')
    parser.add_argument('-yes-cuda', action='store_true', default=True, help='disable the gpu')
    # option
    parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
    parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
    parser.add_argument('-test', action='store_true', default=False, help='train or test')
    parser.add_argument('-xfolds', type=int, default=10, help='number of folds for cross-validation')
    parser.add_argument('-layer-num', type=int, default=2, help='the number of layers in the final MLP')
    parser.add_argument('-word-vector', type=str, default='w2v',
                        help="use of vectors [default: w2v. options: 'glove' or 'w2v' or 'none]")
    parser.add_argument('-emb-path', type=str, default='/fs/project/white.1240/jin/corpora/mikolov_vectors/', help="the path to the word vector file")
    parser.add_argument('-char-vector', type=str, default='none',
                        help="use of vectors [default: none. options: 'char.wiki' or 'none]")
    parser.add_argument('-char-emb-path', type=str, default=os.getcwd(), help="the path to the char vector file")

    parser.add_argument('-min-freq', type=int, default=1, help='minimal frequency to be added to vocab')
    parser.add_argument('-optimizer', type=str, default='adadelta', help="optimizer for all the models [default: SGD. options: 'sgd' or 'adam' or 'adadelta]")
    parser.add_argument('-word-optimizer', type=str, default='adadelta', help="optimizer for all the models [default: SGD. options: 'sgd' or 'adam' or 'adadelta]")
    parser.add_argument('-char-optimizer', type=str, default='adadelta', help="optimizer for all the models [default: SGD. options: 'sgd' or 'adam' or 'adadelta]")
    parser.add_argument('-fine-tune', action='store_true', default=False,
                        help='whether to fine tune the final ensembled model')
    parser.add_argument('-ortho-init', action='store_true', default=False,
                        help='use orthogonalization to improve weight matrix random initialization')
    parser.add_argument('-ensemble', type=str, default='poe',
                        help='ensemble methods [default: poe. options: poe, avg, vot]')
    parser.add_argument('-num-experts', type=int, default=5, help='number of experts if poe is enabled [default: 5]')
    parser.add_argument('-prediction-file-handle', type=str, default='predictions.txt', help='the file to output the test predictions')
    parser.add_argument('-no-always-norm', action='store_true', default=False, help='always max norm the weights')

    # testing parameters
    parser.add_argument('-embed-num', type=int, default=100, help='for testing purposes. do not use.')
    parser.add_argument('-class-num', type=int, default=10, help='for testing purposes. do not use.')
    # end of testing
    args = parser.parse_args()

    if args.word_vector == 'glove':
        args.word_vector = 'glove.6B'
    elif args.word_vector == 'w2v':
        if args.word_embed_dim != 300:
            raise Exception("w2v has no other kind of vectors than 300")
    else:
        args.word_vector = None

    if args.char_vector == 'none':
        args.char_vector = None
    elif args.char_vector == 'char.wiki':
        if args.char_embed_dim != 16:
            raise Exception("char has no other kind of vectors than 16")
    else:
        raise Exception("invalid char embedding name")
    return args