import sys
import os
import time
import torch
import logging
import argparse
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from EfficientNet import EfficientNet
from sampler import WeightedRandom_DistributedSampler
from load_data import valDataProvider, mineral_dataset
from load_data import make_weights_for_unbalanced_classes


logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            datefmt='%m/%d/%Y %H:%M:%S',
            level=logging.INFO)
logger = logging.getLogger(__name__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the
        specified values of k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])
                                            ^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for
                    well-classiﬁed examples (p > .5), putting more focus on
                    hard, misclassiﬁed examples
            size_average(bool): size_average(bool): By default, the losses are
                    averaged over observations for each minibatch. However,
                    if the field size_average is set to False, the losses are
                    instead summed for each minibatch.
    """
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs, dim=1)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        if inputs.is_cuda and not self.alpha.is_cuda:
            if not distributed:
                self.alpha = self.alpha.cuda()
            else:
                self.alpha = self.alpha.to(device)
        alpha = self.alpha[ids.data.view(-1)]
        probs = (P*class_mask).sum(1).view(-1, 1)

        log_p = probs.log()

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


def train_and_val(train_loader, model, cost, optimizer, epochs):

    train_loss_ls = []
    acc1_ls = []

    val_loss_ls = []
    val_acc1_ls = []

    valData = valDataProvider(batch_size*2, val_dataset, distributed)

    for i, (imgs, labels) in enumerate(train_loader):
        # train
        model.train()
        imgs = torch.autograd.Variable(imgs.to(device))
        labels = torch.autograd.Variable(labels.to(device))

        output = model(imgs)
        loss = cost(output, labels)
        train_loss_ls.append(loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        output = F.softmax(output, dim=1)
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        acc1_ls.append(acc1.item())

        if i % 40 == 0:
            print("Loss : {:.3f}  acc1: {:.2f} acc5: {:.2f}".format(
                float(loss), float(acc1), float(acc5)))

        # validate
        model.eval()
        with torch.no_grad():  # avoid out of memory
            val_imgs, val_labels = valData.next()
            val_imgs = torch.autograd.Variable(val_imgs.to(device))
            val_labels = torch.autograd.Variable(val_labels.to(device))

            output = model(val_imgs)
            val_loss = cost(output, val_labels)
            val_loss_ls.append(val_loss.item())

            val_acc1, val_acc5 = accuracy(output, val_labels, topk=(1, 5))
            val_acc1_ls.append(val_acc1.item())

    with open(os.path.join(log_dir, "train_loss.txt"), mode='a') as f:
        f.write(str(train_loss_ls) + '\n')
        f.flush()
    with open(os.path.join(log_dir, "train_acc1.txt"), mode='a') as f:
        f.write(str(acc1_ls) + '\n')
        f.flush()
    with open(os.path.join(log_dir, "val_loss.txt"), mode='a') as f:
        f.write(str(val_loss_ls) + '\n')
        f.flush()
    with open(os.path.join(log_dir, "val_acc1.txt"), mode='a') as f:
        f.write(str(val_acc1_ls) + '\n')
        f.flush()
    ave_train_loss = sum(train_loss_ls) / len(train_loss_ls)
    ave_train_acc1 = sum(acc1_ls) / len(acc1_ls)
    ave_val_acc1 = sum(val_acc1_ls) / len(val_acc1_ls)

    if distributed and args.local_rank != 0:
        return val_acc1_ls
    logger.info('-'*5)
    logger.info("for this epoch")
    logger.info("average train loss : {:.3f}".format(ave_train_loss))
    logger.info("average train acc1 : {:.2f}".format(ave_train_acc1))
    logger.info("average val acc1 : {:.2f}".format(ave_val_acc1))
    return val_acc1_ls


def l_r(epochs):
    return 1e-3 * 0.01 ** (epochs/100)


if __name__ == "__main__":

    if not torch.cuda.is_available():
        print("Error, can't use GPU CUDA.")
        sys.exit()
    else:
        if torch.cuda.device_count() > 1:
            parser = argparse.ArgumentParser()
            parser.add_argument('--local_rank', type=int, default=0,
                                help='node rank for distributed training')
            args = parser.parse_args()

            if args.local_rank == 0:
                logger.info('There are {} GPUs, using distributed'
                            ' training...'.format(torch.cuda.device_count()))

            device = torch.device('cuda:{}'.format(args.local_rank))
            torch.cuda.set_device(args.local_rank)
            torch.distributed.init_process_group(backend="nccl",
                                                 init_method="env://")
            distributed = True
        else:
            logger.info("There is one GPU, using...")
            distributed = False
            device = torch.device('cuda:0')

    nclasses = 36
    batch_size = 32
    epochs = 50

    model = EfficientNet.from_pretrained('efficientnet-b3',
                                         num_classes=nclasses)
    if not distributed:
        model = model.to(device)
    else:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.to(device)
        model = nn.parallel.DistributedDataParallel(
                                                model,
                                                device_ids=[args.local_rank],
                                                output_device=args.local_rank,
                                                find_unused_parameters=True)

    train_dir = "../data_36_classes/for_train/*"
    val_dir = "../data_36_classes/for_val/*"

    train_dataset = mineral_dataset(train_dir, training=True)
    val_dataset = mineral_dataset(val_dir, training=False)

    weights = make_weights_for_unbalanced_classes(train_dataset.imgs,
                                                  nclasses=nclasses)
    weights = torch.DoubleTensor(weights)
    if not distributed:
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights,
                                                                 len(weights))
    else:
        sampler = WeightedRandom_DistributedSampler(train_dataset, weights)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        sampler=sampler,
        drop_last=True)

    log_dir = 'log_for_' + os.path.basename(__file__).split('.py')[0]
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    start = time.time()
    cost = FocalLoss(class_num=nclasses,
                     alpha=(torch.ones(nclasses, 1)*0.25)).to(device)

    acc1_best = 0
    trained_epochs = 0
    for epoch in range(epochs):
        optimzer = torch.optim.Adam(model.parameters(),
                                    lr=l_r(epoch+trained_epochs))
        if distributed and args.local_rank != 0:
            pass
        else:
            logger.info("Epoch {}/{}".format(epoch+1+trained_epochs, epochs))
            logger.info('-'*10)
            logger.info("Training {}th epoch......learning rate : {}".format(
                epoch+1+trained_epochs, l_r(epoch+1+trained_epochs)))

        val_acc1_ls = train_and_val(train_loader,
                                    model, cost,
                                    optimzer,
                                    epochs)
        val_acc1 = sum(val_acc1_ls)/len(val_acc1_ls)

        if distributed and args.local_rank != 0:
            pass
        else:
            time_elapsed = time.time() - start
            logger.info("Used time：{}h {}min".format(
                int(time_elapsed/60//60),
                int(time_elapsed//60 - time_elapsed/60//60 * 60)))

        if val_acc1 > acc1_best:
            if distributed and args.local_rank != 0:
                continue
            else:
                pass
            acc1_best = val_acc1
            logger.info("----------saving model......")
            pth_name = 'epoches_{}_acc1_{:.2f}_%_EfficientNet-b3.pth'\
                .format(epoch+1+trained_epochs, val_acc1)
            torch.save(model.state_dict(), os.path.join(log_dir, pth_name))
            logger.info("ok")
