import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import copy
from chatscript_file_generator import print_test_features

def ensemble_predict(batch, models, args, **kwargs):
    for model in models:
        model.eval()
    logits = []
    confidences = []
    feature, target = batch
    for index, model in enumerate(models):
        logit = model(feature) # log softmaxed
        confidence = model.confidence(feature)
        logits.append(logit)
        confidences.append(confidence)
    total_logit = autograd.Variable(torch.zeros(logits[0].size()))
    average_probs = autograd.Variable(torch.zeros(logits[0].size()))
    average_logprobs = autograd.Variable(torch.zeros(logits[0].size()))
    total_confidence = autograd.Variable(torch.zeros(confidences[0].size()))
    if args.cuda:
        total_logit = total_logit.cuda()
        total_confidence = total_confidence.cuda()
        average_probs = average_probs.cuda()
        average_logprobs = average_logprobs.cuda()

    # calc confidence scores
    for confidence in confidences:
        total_confidence += confidence
    total_confidence /= len(models)

    # calc ensembled prediction
    if args.ensemble == 'poe':
        for some_logit in logits:
            total_logit += some_logit
    elif args.ensemble == 'avg':
        for some_logit in logits:
            total_logit += torch.exp(some_logit)
    elif args.ensemble == 'vot':
        # this does not support backpropagation
        for some_logit in logits:
            _, indices = torch.max(some_logit.data, 1)
            indices.squeeze_()
            # print(indices[:10])
            for index, top_index in enumerate(indices):
                total_logit.data[index,top_index] += 1
    # calc averaged probs
    for some_logit in logits:
        average_probs += torch.exp(some_logit)
    average_probs /= len(models)
    #calc averaged logprobs
    for some_logit in logits:
        average_logprobs += some_logit
    average_logprobs /= len(models)
    # put models back in train mode
    for model in models:
        model.train()

    return total_logit, (total_confidence, average_probs, average_logprobs)

def ensemble_train(trains, devs, models, args, **kwargs):
    print('entering ensemble training:')
    acc_list = []
    for i in range(len(trains)):
        print('ensemble training model {}'.format(i))
        model = models[i]
        if args.cuda:
            model.cuda()
        acc, model = train(trains[i], devs[i], model, args,**kwargs)
        models[i] = model
        acc_list.append(acc)
    return acc_list, models

def ensemble_eval(data_iter, models, args, **kwargs):
    for model in models:
        model.eval()
    corrects, avg_loss = [], []
    logits = []
    data_iter.shuffle = False
    for index, model in enumerate(models):
        for batch in data_iter: # should be only 1 batch
            feature, target = batch.text, batch.label
            feature.data.t_(), target.data.sub_(0)  # batch first, index align
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()

            logit = model(feature) # log softmaxed
            loss = F.nll_loss(logit, target, size_average=False)

            avg_loss = loss.data[0]
            corrects = (torch.max(logit, 1)
                         [1].view(target.size()).data == target.data).sum()

        size = len(data_iter.dataset)
        avg_loss = avg_loss/size
        accuracy = corrects/size * 100.0
        model.train()
        print('\nEvaluation model {} on test - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(index, avg_loss,
                                                                           accuracy,
                                                                           corrects,
                                                                           size))
        if args.verbose:
            print('Evaluation model {} on test - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(index, avg_loss,
                                                                               accuracy,
                                                                               corrects,
                                                                               size), file=kwargs['log_file_handle'])
        logits.append(logit)
    total_logit = 0
    if args.ensemble == 'poe':
        for some_logit in logits:
            # print(some_logit[:10])
            total_logit += some_logit.data
    elif args.ensemble == 'avg':
        total_logit = 0
        for some_logit in logits:
            total_logit += torch.exp(some_logit.data)
    elif args.ensemble == 'vot':
        total_logit = torch.zeros(logits[0].size())
        for some_logit in logits:
            _, indices = torch.max(some_logit.data, 1)
            indices.squeeze_()
            # print(indices[:10])
            for index, top_index in enumerate(indices):
                total_logit[index][top_index] += 1
        if args.cuda:
            total_logit = total_logit.cuda()
    # print(torch.max(total_logit, 1)
    #              [1].view(target.size())[:10])
    # print(target[:10])
    corrects = (torch.max(total_logit, 1)
                 [1].view(target.size()) == target.data).sum()
    size = len(data_iter.dataset)
    accuracy = corrects / size * 100.0
    print('\nEvaluation ensemble {} - acc: {:.4f}%({}/{})'.format(args.ensemble.upper(), accuracy, corrects, size))
    if args.verbose:
        print('Evaluation ensemble {} - acc: {:.4f}%({}/{})'.format(args.ensemble.upper(), accuracy, corrects, size), file=kwargs['log_file_handle'])
    return accuracy

def train(train_iter, dev_iter, model, args, **kwargs):
    if args.cuda:
        model.cuda()
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.l2)
    elif args.optimizer == 'adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(), rho=0.95)
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
            assert feature.volatile is False and target.volatile is False
            # print(feature, target)
            optimizer.zero_grad()
            logit = model(feature)
            loss = F.nll_loss(logit, target)
            loss.backward()
            optimizer.step()

            # max norm constraint
            if args.max_norm > 0:
                if not args.no_always_norm:
                    for row in model.fc1.weight.data:
                        norm = row.norm() + 1e-7
                        row.div_(norm).mul_(args.max_norm)
                else:
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
        loss = F.nll_loss(logit, target, size_average=False)

        avg_loss += loss.data[0]
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()

    size = len(data_iter.dataset)
    avg_loss = avg_loss/size
    accuracy = corrects/size * 100.0
    model.train()
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(avg_loss,
                                                                       accuracy, 
                                                                       corrects, 
                                                                       size))
    if args.verbose:
        print('Evaluation - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(avg_loss,
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

def train_final_ensemble(char_train_data, char_dev_data, word_train_data, word_dev_data, char_model, word_model, last_ensemble_model, args, **kwargs):
    if args.cuda:
        last_ensemble_model.cuda()
    if not args.fine_tune:
        if args.optimizer == 'adam':
            optimizer = torch.optim.Adam(last_ensemble_model.parameters(), lr=args.lr, weight_decay=args.l2)
        elif args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(last_ensemble_model.parameters(), lr=args.lr, weight_decay=args.l2)
        elif args.optimizer == 'adadelta':
            optimizer = torch.optim.Adadelta(last_ensemble_model.parameters(), lr=args.lr, weight_decay=args.l2, rho=0.95)
        else:
            raise Exception("bad optimizer!")
    else:
        raise NotImplementedError('fine tuning is not implemented')
        # char_cnn_params = {'params':char_model.parameters(), 'lr':args.lr * 1e-3}
        # word_cnn_params = {'params':word_model.parameters(), 'lr':args.lr * 1e-3}
        # logistic_params = {'params':logistic_model.parameters()}
        # optimizer = torch.optim.SGD([char_cnn_params, word_cnn_params, logistic_params], lr=args.lr)

    steps = 0
    last_ensemble_model.train()

    for epoch in range(1, args.epochs+1):
        for char_batch, word_batch in zip(char_train_data,word_train_data):
            char_feature, char_target = char_batch.text, char_batch.label
            char_feature.data.t_()
            word_feature, word_target = word_batch.text, word_batch.label
            word_feature.data.t_()

            if args.cuda:
                char_feature, char_target = char_feature.cuda(), char_target.cuda()
                word_feature, word_target = word_feature.cuda(), word_target.cuda()

            assert torch.equal(char_target.data, word_target.data), "Mismatching data sample! {}, {}".format(char_target.data,
                                                                                                        word_target.data)
            if args.num_experts == 0:
                char_output = char_model(char_feature)
                word_output = word_model(word_feature)
            else:
                char_train_tensors = (char_feature, char_target)
                word_train_tensors = (word_feature, word_target)
                char_output, _ = ensemble_predict(char_train_tensors, char_model, args)
                word_output, _ = ensemble_predict(word_train_tensors, word_model, args)

            if not args.fine_tune:
                char_output = autograd.Variable(char_output.data)
                word_output = autograd.Variable(word_output.data)

            optimizer.zero_grad()
            logit = last_ensemble_model((char_output, word_output))
            loss = F.nll_loss(logit, char_target)
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
                eval_final_ensemble(char_dev_data, word_dev_data, char_model, word_model, last_ensemble_model, args, **kwargs)
            # if steps % args.save_interval == 0:
            #     if not os.path.isdir(args.save_dir): os.makedirs(args.save_dir)
            #     save_prefix = os.path.join(args.save_dir, 'snapshot')
            #     save_path = '{}_steps{}.pt'.format(save_prefix, steps)
            #     torch.save(logistic_model, save_path)
    acc = eval_final_ensemble(char_dev_data, word_dev_data, char_model, word_model, last_ensemble_model, args, **kwargs)
    return acc

def eval_final_ensemble(char_data, word_data, char_model, word_model, last_ensemble_model, args, **kwargs):
    last_ensemble_model.eval()
    corrects, avg_loss = 0, 0
    for char_batch, word_batch in zip(char_data, word_data):
        char_feature, char_target = char_batch.text, char_batch.label
        char_feature.data.t_()  # batch first, index align
        char_feature.volatile = True

        word_feature, word_target = word_batch.text, word_batch.label
        word_feature.data.t_() # batch first, index align
        word_feature.volatile = True

        if args.cuda:
            char_feature, char_target = char_feature.cuda(), char_target.cuda()
            word_feature, word_target = word_feature.cuda(), word_target.cuda()
        assert torch.equal(char_target.data, word_target.data), "Mismatching data sample! {}, {}".format(char_target.data, word_target.data)

        if args.num_experts == 0:
            char_output = char_model(char_feature)
            word_output = word_model(word_feature)
        else:
            char_tensors = (char_feature, char_target)
            word_tensors = (word_feature, word_target)
            char_output, (char_confidence, char_ave_probs, char_ave_logprobs) = ensemble_predict(char_tensors, char_model, args)
            word_output, (word_confidence, word_ave_probs, word_ave_logprobs) = ensemble_predict(word_tensors, word_model, args)
        # print(char_output)
        # print(word_output)
        logit = last_ensemble_model((char_output, word_output))
        loss = F.nll_loss(logit, char_target, size_average=False)

        char_feature.volatile = False
        word_feature.volatile = False

        avg_loss += loss.data[0]
        corrects += (torch.max(logit, 1)
                     [1].view(char_target.size()).data == char_target.data).sum()
        if 'prediction_file_handle' in kwargs:
            print_test_features(logit, char_confidence+word_confidence,char_ave_probs+word_ave_probs, char_ave_logprobs+word_ave_logprobs, char_target, kwargs['dialogues'], kwargs['labels'], kwargs['indices'],
                                kwargs['fold_id'], kwargs['chats'], kwargs['prediction_file_handle'])
    size = len(char_data.data())
    avg_loss = avg_loss / size
    accuracy = corrects / size * 100.0
    last_ensemble_model.train()
    print('\nEvaluation last ensemble - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                       size))
    if args.verbose:
        print('Evaluation last ensemble - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                           accuracy,
                                                                           corrects,
                                                                           size), file=kwargs['log_file_handle'])
    return accuracy

