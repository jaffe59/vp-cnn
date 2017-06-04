import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import copy

def train(train_iter, dev_iter, model, args, **kwargs):
    if args.cuda:
        model.cuda()
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.l2)
    elif args.optimizer == 'adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr, weight_decay=args.l2, rho=0.95)
    else:
        raise Exception("bad optimizer!")

    steps = 0
    model.train()
    best_acc = 0
    best_model = None
    for epoch in range(1, args.epochs+1):
        for batch in train_iter:
            feature, target = batch.text, batch.label
            feature.data.t_(), target.data.sub_(0)  # batch first, index align
            # print(feature)
            # print(train_iter.data().fields['text'].vocab.stoi)
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()
            # print(feature, target)
            optimizer.zero_grad()
            logit = model(feature)
            loss = F.cross_entropy(logit, target)
            loss.backward()
            optimizer.step()

            # max norm constraint
            if args.max_norm > 0:
                model.fc1.weight.data.renorm_(2, 0, args.max_norm)

            steps += 1
            if steps % args.log_interval == 0:
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = corrects/batch.batch_size * 100.0
                sys.stdout.write(
                    '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps, 
                                                                             loss.data[0], 
                                                                             accuracy,
                                                                             corrects,
                                                                             batch.batch_size))
            if steps % args.test_interval == 0:
                if args.verbose:
                    corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                    accuracy = corrects/batch.batch_size * 100.0
                    print(
                    'Batch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps,
                                                                             loss.data[0],
                                                                             accuracy,
                                                                             corrects,
                                                                             batch.batch_size), file=kwargs['log_file_handle'])
        acc = eval(dev_iter, model, args, **kwargs)
        if acc > best_acc:
            best_acc = acc
            best_model = copy.deepcopy(model)
        # print(model.embed.weight[100])
            # if steps % args.save_interval == 0:
            #     if not os.path.isdir(args.save_dir): os.makedirs(args.save_dir)
            #     save_prefix = os.path.join(args.save_dir, 'snapshot')
            #     save_path = '{}_steps{}.pt'.format(save_prefix, steps)
            #     torch.save(model, save_path)
    model = best_model
    acc = eval(dev_iter, model, args, **kwargs)
    return acc, model

def eval(data_iter, model, args, **kwargs):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target = batch.text, batch.label
        feature.data.t_(), target.data.sub_(0)  # batch first, index align
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()

        logit = model(feature)
        loss = F.cross_entropy(logit, target, size_average=False)

        avg_loss += loss.data[0]
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()

    size = len(data_iter.dataset)
    avg_loss = avg_loss/size
    accuracy = corrects/size * 100.0
    model.train()
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss, 
                                                                       accuracy, 
                                                                       corrects, 
                                                                       size))
    if args.verbose:
        print('Evaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                           accuracy,
                                                                           corrects,
                                                                           size), file=kwargs['log_file_handle'])
    return accuracy

def predict(text, model, text_field, label_feild):
    assert isinstance(text, str)
    model.eval()
    text = text_field.tokenize(text)
    text = text_field.preprocess(text)
    text = [[text_field.vocab.stoi[x] for x in text]]
    x = text_field.tensor_type(text)
    x = autograd.Variable(x, volatile=True)
    print(x)
    output = model(x)
    _, predicted = torch.max(output, 1)
    return label_feild.vocab.itos[predicted.data[0][0]+1]

def train_logistic(char_train_data, char_dev_data, word_train_data, word_dev_data, char_model, word_model, logistic_model, args, **kwargs):
    if args.cuda:
        logistic_model.cuda()
    if not args.fine_tune:
        if args.optimizer == 'adam':
            optimizer = torch.optim.Adam(logistic_model.parameters(), lr=args.lr, weight_decay=args.l2)
        elif args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(logistic_model.parameters(), lr=args.lr, weight_decay=args.l2)
        elif args.optimizer == 'adadelta':
            optimizer = torch.optim.Adadelta(logistic_model.parameters(), lr=args.lr, weight_decay=args.l2, rho=0.95)
        else:
            raise Exception("bad optimizer!")
    else:
        char_cnn_params = {'params':char_model.parameters(), 'lr':args.lr * 1e-3}
        word_cnn_params = {'params':word_model.parameters(), 'lr':args.lr * 1e-3}
        logistic_params = {'params':logistic_model.parameters()}
        optimizer = torch.optim.SGD([char_cnn_params, word_cnn_params, logistic_params], lr=args.lr)

    steps = 0
    logistic_model.train()
    if not args.fine_tune:
        char_model.eval()
        word_model.eval()
    for epoch in range(1, args.epochs+1):
        for char_batch, word_batch in zip(char_train_data,word_train_data):
            # word_batch = next(word_train_data)

            char_feature, char_target = char_batch.text, char_batch.label
            char_feature.data.t_(), char_target.data.sub_(1)  # batch first, index align

            word_feature, word_target = word_batch.text, word_batch.label
            word_feature.data.t_(), word_target.data.sub_(1)  # batch first, index align
            # print(char_batch.data, word_batch.data)
            if args.cuda:
                char_feature, char_target = char_feature.cuda(), char_target.cuda()
                word_feature, word_target = word_feature.cuda(), word_target.cuda()

            assert torch.equal(char_target.data, word_target.data), "Mismatching data sample! {}, {}".format(char_target.data,
                                                                                                        word_target.data)

            char_output = char_model(char_feature)
            word_output = word_model(word_feature)

            if not args.fine_tune:
                char_output = autograd.Variable(char_output.data)
                word_output = autograd.Variable(word_output.data)

            optimizer.zero_grad()
            logit = logistic_model(char_output, word_output)
            loss = F.cross_entropy(logit, char_target)
            loss.backward()
            optimizer.step()

            steps += 1
            if steps % args.log_interval == 0:
                corrects = (torch.max(logit, 1)[1].view(char_target.size()).data == char_target.data).sum()
                accuracy = corrects/char_batch.batch_size * 100.0
                sys.stdout.write(
                    '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps,
                                                                             loss.data[0],
                                                                             accuracy,
                                                                             corrects,
                                                                             char_batch.batch_size))
            if steps % args.test_interval == 0:
                if args.verbose:
                    corrects = (torch.max(logit, 1)[1].view(char_target.size()).data == char_target.data).sum()
                    accuracy = corrects / char_batch.batch_size * 100.0
                    print(
                    'Batch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps,
                                                                             loss.data[0],
                                                                             accuracy,
                                                                             corrects,
                                                                           char_batch.batch_size), file=kwargs['log_file_handle'])
                eval_logistic(char_dev_data, word_dev_data, char_model, word_model, logistic_model, args, **kwargs)
            # if steps % args.save_interval == 0:
            #     if not os.path.isdir(args.save_dir): os.makedirs(args.save_dir)
            #     save_prefix = os.path.join(args.save_dir, 'snapshot')
            #     save_path = '{}_steps{}.pt'.format(save_prefix, steps)
            #     torch.save(logistic_model, save_path)
    acc = eval_logistic(char_dev_data, word_dev_data, char_model, word_model, logistic_model, args, **kwargs)
    return acc

def eval_logistic(char_data, word_data, char_model, word_model, logistic_model, args, **kwargs):
    logistic_model.eval()
    corrects, avg_loss = 0, 0
    word_model.eval()
    char_model.eval()
    for char_batch, word_batch in zip(char_data, word_data):
        char_feature, char_target = char_batch.text, char_batch.label
        char_feature.data.t_(), char_target.data.sub_(1)  # batch first, index align
        char_feature.volatile = True

        word_feature, word_target = word_batch.text, word_batch.label
        word_feature.data.t_(), word_target.data.sub_(1)  # batch first, index align
        word_feature.volatile = True

        if args.cuda:
            char_feature, char_target = char_feature.cuda(), char_target.cuda()
            word_feature, word_target = word_feature.cuda(), word_target.cuda()
        assert torch.equal(char_target.data, word_target.data), "Mismatching data sample! {}, {}".format(char_target.data, word_target.data)

        char_output = char_model(char_feature)
        word_output = word_model(word_feature)


        logit = logistic_model(char_output, word_output)
        loss = F.cross_entropy(logit, char_target, size_average=False)

        char_feature.volatile = False
        word_feature.volatile = False

        avg_loss += loss.data[0]
        corrects += (torch.max(logit, 1)
                     [1].view(char_target.size()).data == char_target.data).sum()

    size = len(char_data.data())
    avg_loss = avg_loss / size
    accuracy = corrects / size * 100.0
    logistic_model.train()
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                       size))
    if args.verbose:
        print('Evaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                           accuracy,
                                                                           corrects,
                                                                           size), file=kwargs['log_file_handle'])
    return accuracy

