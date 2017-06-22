import torch
import torch.nn.functional as F
import sys
import copy
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
    # objf = torch.nn.CosineEmbeddingLoss()
    dev_loss = 0
    train_loss = 0
    for epoch in range(1, args.epochs+1):
        for batch in train_iter:
            s1, s2, target = batch.s1, batch.s2, batch.label
            s1.data.t_(), s2.data.t_() # batch first, index align
            _, index = torch.max(batch.label, 0)
            print(index)
            # print(feature)
            # print(train_iter.data().fields['text'].vocab.stoi)

            assert s1.volatile is False and target.volatile is False
            # print(feature, target)
            optimizer.zero_grad()
            sim_score = model((s1, s2))
            # loss = objf(sim_score[0], sim_score[1], target)
            y = F.cosine_similarity(sim_score[0], sim_score[1]).view(1, -1)
            y = F.log_softmax(y)
            print(y)
            loss = F.nll_loss(y, index)
            loss.backward()
            optimizer.step()
            train_loss += loss.data[0]
            # max norm constraint
            if args.max_norm > 0:
                if not args.no_always_norm:
                    for row in model.cnn.fc1.weight.data:
                        norm = row.norm() + 1e-7
                        row.div_(norm).mul_(args.max_norm)
                else:
                    model.cnn.fc1.weight.data.renorm_(2, 0, args.max_norm)

            steps += 1
        else:
            sys.stdout.write(
                    '\rEpoch {} - loss: {:.6f}  {})'.format(epoch,train_loss/len(train_iter.dataset),args.batch_size))
            train_loss = 0
            # if steps % args.test_interval == 0:
            #     if args.verbose:
            #         corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
            #         accuracy = corrects/batch.batch_size * 100.0
            #         print(
            #         'Batch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps,
            #                                                                  loss.data[0],
            #                                                                  accuracy,
            #                                                                  corrects,
            #                                                                  batch.batch_size), file=kwargs['log_file_handle'])
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
    total_loss = 0
    objf = torch.nn.CosineEmbeddingLoss()
    for batch in data_iter:
        s1, s2, target = batch.s1, batch.s2, batch.label
        s1.data.t_(), s2.data.t_()  # batch first, index align

        assert s1.volatile is True
        # print(feature, target)
        sim_score = model((s1, s2))
        loss = objf(sim_score[0], sim_score[1], target)
        total_loss += loss.data[0]

    size = len(data_iter.dataset)
    avg_loss = total_loss/size
    model.train()
    print('\nEvaluation - loss: {:.6f} )'.format(avg_loss))
    if args.verbose:
        print('Evaluation - loss: {:.6f} )'.format(avg_loss), file=kwargs['log_file_handle'])
    return total_loss
