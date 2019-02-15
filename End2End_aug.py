import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# import matplotlib.pyplot as plt
import time
import torchvision.transforms as transforms
import torchvision.models.resnet as resnet
import resnet_v2
import PIL.Image as Image
import utils
import math
import copy
import os

dataset = 'cifar100'
images_train, labels_train, images_val, labels_val, images_test, labels_test = utils.load_data(dataset)


# parameters
iteration = 400
iteration_finetune = 300
lr = 0.5
schedules = range(50, iteration, 50)
gamma = 0.5
momentum = 0.9
decay = 0.0001
# decay = 0.0005
batchsize = 400
num_class = 100
num_class_novel = 10
memory_K = 2000
T = 2
dist_ratio = 0.5
gredient_noise_ratio = 0
network_depth = 32
flag_augmentation_e2e = True
stop_acc = 0.998
flag_dist_all = False


period_train = num_class//num_class_novel
memory_size = memory_K//num_class_novel

net = resnet_v2.resnet(depth=network_depth, num_classes=100)

# print parameters
print('parameters:')
print('dataset: ', dataset)
print('lr: ', lr)
# print('schedules: follow the paper, divide 10 per 10 steps')
print('schedules: ', schedules)
print('gamma: ', gamma)
print('momentum: ', momentum)
print('decay: ', decay)
print('batchsize: ', batchsize)
print('iteration: ', iteration)
print('iteration_finetune: ', iteration_finetune)
print('num_class: ', num_class)
print('num_class_novel: ', num_class_novel)
print('memory_K: ', memory_K)
print('T: ', T)
print('dist_ratio: ', dist_ratio)
print('gredient_noise_ratio: ', gredient_noise_ratio)
print('network_depth: ', network_depth)
print('flag_augmentation_e2e: ', flag_augmentation_e2e)
print('stop_acc: ', stop_acc)
print('flag_dist_all: ', flag_dist_all)


# gpu
num_gpu = torch.cuda.device_count()
if num_gpu > 0:
    print('GPU number = %d' % (num_gpu))
    device_ids = np.arange(num_gpu).tolist()
    print('device_ids:')
    print(device_ids)
    net = nn.DataParallel(net, device_ids=device_ids).cuda()
else:
    print('only cpu is available')



np.random.seed(100)
class_order = np.random.permutation(num_class)

print('class order:')
print(class_order)
class_old = np.array([], dtype=int)
memory_images = np.zeros(shape=(0, memory_size, 3, 32, 32), dtype=np.uint8)
memory_labels = np.zeros(shape=(0, memory_size), dtype=int)
acc_nvld_basic = np.zeros((period_train))
acc_nvld_finetune = np.zeros((period_train))
crossentropy = nn.CrossEntropyLoss()

# get feature dim
feat = net.forward(torch.from_numpy(np.zeros(shape=(1, 3, 32, 32))).float().cuda())
dim = np.shape(feat.cpu().data.numpy())[-1]
print('feature dim = %d'%(dim))

# get model state
first_model_path = 'model/first_model_e2e_aug_%d_%d_%s%s' % (network_depth, 0, ''.join(str(e) for e in class_order[:num_class_novel]), '.pkl')
flag_model = os.path.exists(first_model_path)


for period in range(period_train):
    print('------------------')
    print('------------------')
    print('period = %d'%(period))
    class_novel = class_order[period*num_class_novel:(period+1)*num_class_novel]
    print('class_novel:')
    print(class_novel)
    images_novel_train = images_train[class_novel]
    images_novel_train = np.reshape(images_novel_train, newshape=(-1, 3, 32, 32))
    labels_novel_train = labels_train[class_novel]
    labels_novel_train = np.reshape(labels_novel_train, newshape=(-1))
    images_novel_test = images_test[class_novel]
    images_novel_test = np.reshape(images_novel_test, newshape=(-1, 3, 32, 32))
    labels_novel_test = labels_test[class_novel]
    labels_novel_test = np.reshape(labels_novel_test, newshape=(-1))

    num_class_old = class_old.shape[0]
    if period == 0:
        images_combined_train = images_novel_train
        labels_combined_train = labels_novel_train
    else:
        images_combined_train = np.concatenate((images_novel_train, np.reshape(memory_images, newshape=(-1, 3, 32, 32))), axis=0)
        labels_combined_train = np.concatenate((labels_novel_train, np.reshape(memory_labels, newshape=(-1))), axis=0)


    images_nvld_test = images_test[np.concatenate((class_old, class_novel), axis=0)]
    images_nvld_test = np.reshape(images_nvld_test, newshape=(-1, 3, 32, 32))
    labels_nvld_test = labels_test[np.concatenate((class_old, class_novel), axis=0)]
    labels_nvld_test = np.reshape(labels_nvld_test, newshape=(-1))


    # data augmentation
    if flag_augmentation_e2e == True:
        # augmentation
        images_combined_train, labels_combined_train = utils.data_augmentation_e2e(images_combined_train,
                                                                                   labels_combined_train)

        # normalization
        images_combined_train = images_combined_train/255.0
        images_nvld_test = images_nvld_test/255.0

        v_mean_0 = np.mean(images_train[:,:,0,:,:]/255.0)
        v_mean_1 = np.mean(images_train[:,:,1,:,:]/255.0)
        v_mean_2 = np.mean(images_train[:,:,2,:,:]/255.0)

        images_combined_train[:,0] -= v_mean_0
        images_combined_train[:,1] -= v_mean_1
        images_combined_train[:,2] -= v_mean_2
        images_nvld_test[:, 0] -= v_mean_0
        images_nvld_test[:, 1] -= v_mean_1
        images_nvld_test[:, 2] -= v_mean_2



    print('training size = %d'%(labels_combined_train.shape[0]))

    # training

    lrc = lr

    print('current lr = %f' % (lrc))
    acc_training = []
    softmax = nn.Softmax(dim=-1).cuda()


    ##################################
    net_old = copy.deepcopy(net)
    ##################################


    for iter in range(iteration):

        # learning rate
        if iter in schedules:
            lrc *= gamma
            print('current lr = %f'%(lrc))

        # criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(net.parameters(), lr=lrc, momentum=momentum,
                                    weight_decay=decay, nesterov=True)



        # train
        idx_train = np.random.permutation(labels_combined_train.shape[0])
        loss_avg = 0
        loss_cls_avg = 0
        loss_dist_avg = 0
        acc_avg = 0
        num_exp = 0
        tstart = time.clock()
        batchnum_train = math.ceil(labels_combined_train.shape[0] / batchsize)

        # load model
        if period == 0 and flag_model:
            print('load model: %s' % first_model_path)
            net.load_state_dict(torch.load(first_model_path))

            # break


        for bi in range(batchnum_train):

            if period == 0 and flag_model: # loaded model, do not need training
                num_exp = 1
                break

            if bi == batchnum_train - 1:
                idx = idx_train[bi * batchsize:]
            else:
                idx = idx_train[bi * batchsize:(bi + 1) * batchsize]
            img = images_combined_train[idx]
            lab = labels_combined_train[idx]
            lab_onehot = utils.one_hot(lab, num_class)

            # transform
            if flag_augmentation_e2e == False:  # old transform
                img = utils.img_transform(img, 'train')
            img = torch.from_numpy(img).float()
            img = img.cuda()
            lab_onehot = torch.from_numpy(lab_onehot)
            lab_onehot = lab_onehot.float()

            lab_onehot = lab_onehot.cuda()

            # print("Outside: input size", img.size(), "output_size", lab.size())

            output = net.forward(img)


            # classification loss
            indices = torch.LongTensor(np.concatenate((class_old, class_novel), axis=0))
            indices = indices.cuda()
            prob_cls = torch.index_select(output, 1, indices)
            prob_cls = softmax(prob_cls)
            lab_onehot = torch.index_select(lab_onehot, 1, indices)

            loss_cls = F.binary_cross_entropy(prob_cls, lab_onehot)

            # distillation loss for only old class data !!!
            if period>0:
                indices = torch.LongTensor(class_old)
                indices = indices.cuda()
                dist = torch.index_select(output, 1, indices)
                dist = softmax(dist/T)

                output_old = net_old.forward(img)
                output_old = torch.index_select(output_old, 1, indices)
                lab_dist = Variable(output_old, requires_grad = False)
                lab_dist = softmax(lab_dist/T)

                if not flag_dist_all:
                    # only for old class data
                    indices = [id for id, la in enumerate(lab) if la in class_old]
                    indices = torch.LongTensor(indices)
                    indices = indices.cuda()
                    dist = torch.index_select(dist, 0, indices)
                    lab_dist = torch.index_select(lab_dist, 0, indices)

                loss_dist = F.binary_cross_entropy(dist, lab_dist)
            else:
                loss_dist = 0

            loss = loss_cls + dist_ratio*loss_dist

            loss_avg += loss.item()
            loss_cls_avg += loss_cls.item()
            if period == 0:
                loss_dist_avg += 0
            else:
                loss_dist_avg += loss_dist.item()

            acc = np.sum(np.equal(np.argmax(prob_cls.cpu().data.numpy(), axis=-1), np.argmax(lab_onehot.cpu().data.numpy(), axis=-1)))
            acc_avg += acc
            num_exp += np.shape(lab)[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()



            # add random noise to gradients / weights
            if gredient_noise_ratio > 0:
                for p in net.parameters():
                    p.data.sub_(gredient_noise_ratio * lrc * torch.from_numpy(
                        (np.random.random(np.shape(p.data.cpu().data.numpy())) - 0.5)*2).float().cuda())


        loss_avg /= num_exp
        loss_cls_avg /= num_exp
        loss_dist_avg /= num_exp
        acc_avg /= num_exp
        acc_training.append(acc_avg)
        tend = time.clock()
        tcost = tend - tstart

        print('Training Period: %d \t Iter: %d \t time = %.1f \t loss = %.6f \t acc = %.4f' % (period, iter, tcost, loss_avg, acc_avg))

        # print('Training Period: %d \t Iter: %d \t time = %.1f \t loss_cls = %.6f \t loss_dist = %.6f \t loss = %.6f \t acc = %.4f'%(period, iter, tcost, loss_cls_avg, loss_dist_avg, loss_avg, acc_avg))



        # test all (novel + old) classes based on logists
        if period > -1:
            # images_nvld_test = images_test[np.concatenate((class_old, class_novel), axis=0)]
            # images_nvld_test = np.reshape(images_nvld_test, newshape=(-1, 3, 32, 32))
            # labels_nvld_test = labels_test[np.concatenate((class_old, class_novel), axis=0)]
            # labels_nvld_test = np.reshape(labels_nvld_test, newshape=(-1))
            idx_test = np.random.permutation(labels_nvld_test.shape[0])
            loss_avg = 0
            acc_avg = 0
            num_exp = 0
            tstart = time.clock()

            batchnum_test = math.ceil(labels_nvld_test.shape[0] / batchsize)
            for bi in range(batchnum_test):
                if bi == batchnum_test - 1:
                    idx = idx_test[bi * batchsize:]
                else:
                    idx = idx_test[bi * batchsize:(bi + 1) * batchsize]
                img = images_nvld_test[idx]
                lab = labels_nvld_test[idx]
                lab_onehot = utils.one_hot(lab, num_class)

                # normalization
                if flag_augmentation_e2e == False:  # old transform
                    img = utils.img_transform(img, 'test')
                img = torch.from_numpy(img).float()
                img = img.cuda()
                output = net.forward(img)
                indices = torch.LongTensor(np.concatenate((class_old, class_novel), axis=0))
                indices = indices.cuda()
                output = torch.index_select(output, 1, indices)
                output = softmax(output)
                output = output.cpu().data.numpy()
                lab_onehot = lab_onehot[:, np.concatenate((class_old, class_novel), axis=0)]

                acc = np.sum(np.equal(np.argmax(output, axis=-1), np.argmax(lab_onehot, axis=-1)))
                acc_avg += acc
                num_exp += np.shape(lab)[0]

            acc_avg /= num_exp
            tend = time.clock()
            tcost = tend - tstart
            print('Testing novel+old Period: %d \t Iter: %d \t time = %.1f \t\t\t\t\t\t acc = %.4f' % (period, iter, tcost, acc_avg))
            acc_nvld_basic[period] = acc_avg

        if period == 0 and flag_model: # loaded model, do not need extra test
            break

        if len(acc_training)>20 and acc_training[-1]>stop_acc and acc_training[-5]>stop_acc:
            print('training loss converged')
            break


    # save model
    if period == 0 and (not flag_model):
        print('save model: %s' % first_model_path)
        torch.save(net.state_dict(), first_model_path)


    # balanced finetune

    net_old = copy.deepcopy(net)

    # finetune
    if period>0:
        # lrc = lr
        lrc = lr*0.1 # small learning rate for finetune
        print('current lr = %f' % (lrc))
        softmax = nn.Softmax(dim=-1).cuda()

        acc_finetune_train = []
        for iter in range(iteration_finetune):

            # learning rate
            if iter in schedules:
                lrc *= gamma
                print('current lr = %f'%(lrc))

            # criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(net.parameters(), lr=lrc, momentum=momentum,
                                        weight_decay=decay, nesterov=True)



            # finetune train
            idx_finetune_novel = np.random.permutation(labels_novel_train.shape[0])
            idx_finetune_novel = idx_finetune_novel[:memory_size*num_class_novel]
            idx_finetune_old = np.arange(start=labels_novel_train.shape[0], stop=labels_combined_train.shape[0])
            idx_finetune = np.concatenate((idx_finetune_novel, idx_finetune_old), axis=0)
            np.random.shuffle(idx_finetune)


            loss_avg = 0
            acc_avg = 0
            num_exp = 0
            tstart = time.clock()



            batchnum_train = math.ceil(idx_finetune.shape[0] // batchsize)
            for bi in range(batchnum_train):
                if bi == batchnum_train - 1:
                    idx = idx_finetune[bi * batchsize:]
                else:
                    idx = idx_finetune[bi * batchsize:(bi + 1) * batchsize]
                img = images_combined_train[idx]
                lab = labels_combined_train[idx]
                lab_onehot = utils.one_hot(lab, num_class)

                # transform
                if flag_augmentation_e2e == False:
                    img = utils.img_transform(img, 'train')
                img = torch.from_numpy(img).float()
                img = img.cuda()
                lab_onehot = torch.from_numpy(lab_onehot)
                lab_onehot = lab_onehot.float()
                lab_onehot = lab_onehot.cuda()

                # print("Outside: input size", img.size(), "output_size", lab.size())
                output = net.forward(img)

                # classification loss
                indices = torch.LongTensor(np.concatenate((class_old, class_novel), axis=0))
                indices = indices.cuda()
                prob_cls = torch.index_select(output, 1, indices)
                prob_cls = softmax(prob_cls)
                lab_onehot = torch.index_select(lab_onehot, 1, indices)
                loss_cls = F.binary_cross_entropy(prob_cls, lab_onehot)

                # distillation loss for all classes (maybe the author only distillates for novel classes)
                if period>0:
                    indices = torch.LongTensor(np.concatenate((class_old, class_novel), axis=0))
                    indices = indices.cuda()
                    dist = torch.index_select(output, 1, indices)
                    dist = softmax(dist/T)


                    output_old = net_old.forward(img)
                    output_old = torch.index_select(output_old, 1, indices)
                    lab_dist = Variable(output_old, requires_grad=False)



                    lab_dist = softmax(lab_dist/T)
                    loss_dist = F.binary_cross_entropy(dist, lab_dist)
                else:
                    loss_dist = 0

                loss = loss_cls + dist_ratio*loss_dist

                loss_avg += loss.item()
                acc = np.sum(np.equal(np.argmax(prob_cls.cpu().data.numpy(), axis=-1), np.argmax(lab_onehot.cpu().data.numpy(), axis=-1)))
                acc_avg += acc
                num_exp += np.shape(lab)[0]

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # add random noise to gradients / weights
                if gredient_noise_ratio > 0:
                    for p in net.parameters():
                        p.data.sub_(gredient_noise_ratio * lrc * torch.from_numpy(
                            (np.random.random(np.shape(p.data.cpu().data.numpy())) - 0.5) * 2).float().cuda())

            loss_avg /= num_exp
            acc_avg /= num_exp
            acc_finetune_train.append(acc_avg)
            tend = time.clock()
            tcost = tend - tstart
            print('Finetune Training Iter: %d \t time = %.1f \t loss = %.6f \t acc = %.4f'%(iter, tcost, loss_avg, acc_avg))


            # test all (novel + old) classes based on logists
            if period > 0:
                # images_nvld_test = images_test[np.concatenate((class_old, class_novel), axis=0)]
                # images_nvld_test = np.reshape(images_nvld_test, newshape=(-1, 3, 32, 32))
                # labels_nvld_test = labels_test[np.concatenate((class_old, class_novel), axis=0)]
                # labels_nvld_test = np.reshape(labels_nvld_test, newshape=(-1))
                idx_test = np.random.permutation(labels_nvld_test.shape[0])
                loss_avg = 0
                acc_avg = 0
                num_exp = 0
                tstart = time.clock()

                batchnum_test = math.ceil(labels_nvld_test.shape[0] // batchsize)
                for bi in range(batchnum_test):
                    if bi == batchnum_test - 1:
                        idx = idx_test[bi * batchsize:]
                    else:
                        idx = idx_test[bi * batchsize:(bi + 1) * batchsize]
                    img = images_nvld_test[idx]
                    lab = labels_nvld_test[idx]
                    lab_onehot = utils.one_hot(lab, num_class)

                    if flag_augmentation_e2e == False:  # old transform
                        img = utils.img_transform(img, 'test')
                    img = torch.from_numpy(img).float()
                    img = img.cuda()

                    # # normalization
                    output = net.forward(img)
                    indices = torch.LongTensor(np.concatenate((class_old, class_novel), axis=0))
                    indices = indices.cuda()
                    output = torch.index_select(output, 1, indices)
                    output = softmax(output)
                    output = output.cpu().data.numpy()
                    lab_onehot = lab_onehot[:, np.concatenate((class_old, class_novel), axis=0)]

                    acc = np.sum(np.equal(np.argmax(output, axis=-1), np.argmax(lab_onehot, axis=-1)))
                    acc_avg += acc
                    num_exp += np.shape(lab)[0]

                acc_avg /= num_exp
                tend = time.clock()
                tcost = tend - tstart
                print('Finetune Testing novel+old Period: %d \t Iter: %d \t time = %.1f \t\t\t\t\t\t acc = %.4f' % (period, iter, tcost, acc_avg))
                acc_nvld_finetune[period] = acc_avg

            if len(acc_finetune_train) > 20 and acc_finetune_train[-1] > stop_acc and acc_finetune_train[-5] > stop_acc:
                print('training loss converged')
                break

    if period>0:
        print('------------------- result ------------------------')
        print('Period: %d, basic acc = %.4f, finetune acc = %.4f' % (period, acc_nvld_basic[period], acc_nvld_finetune[period]))
        print('---------------------------------------------------')

    if period == period_train-1:
        print('------------------- ave result ------------------------')
        print('basic acc = %.4f, finetune acc = %.4f' % (np.mean(acc_nvld_basic[1:], keepdims=False), np.mean(acc_nvld_finetune[1:], keepdims=False)))
        print('---------------------------------------------------')



    # memory management [random select]
    # reduce old memory
    memory_size = memory_K // (num_class_novel + num_class_old)
    if period >0:
        memory_images = memory_images[:, :memory_size]
        memory_labels = memory_labels[:, :memory_size]
    # add new memory
    memory_new_images = np.zeros((num_class_novel, memory_size, 3, 32, 32), dtype=np.uint8)
    memory_new_labels = np.zeros((num_class_novel, memory_size), dtype=np.int32)
    # random selection
    for k in range(num_class_novel):

        img = images_train[class_novel[k], np.random.permutation(500)[:memory_size]]
        memory_new_images[k] = img
        memory_new_labels[k] = np.tile(class_novel[k], (memory_size))

    # herding
    # to do

    if period > 0:
        memory_images = np.concatenate((memory_images, memory_new_images), axis=0)
        memory_labels = np.concatenate((memory_labels, memory_new_labels), axis=0)
    else:
        memory_images = memory_new_images
        memory_labels = memory_new_labels


    class_old = np.append(class_old, class_novel, axis=0)





print('xxx')