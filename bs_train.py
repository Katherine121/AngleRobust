import argparse
import math
import os
import random
import shutil
import time
import warnings
import builtins
import torch.distributed as dist
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
from torch import nn
import torchvision.transforms as transforms
import torchvision.models as torchvision_models
from torchvision.models import MobileNet_V3_Small_Weights
from thop import profile

from baseline.bs_models import LSTM
from datasets import OrderTrainDataset, OrderTestDataset
from utils import UncertaintyLoss

torch.set_printoptions(precision=8)

parser = argparse.ArgumentParser(description='PyTorch Training')

parser.add_argument('--start-epoch', default=0, type=int,
                    metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', default=120, type=int,
                    metavar='N', help='number of total epochs to run')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='FREQ', help='print frequency (default: 10)')

parser.add_argument('--save-dir', default='baseline/moblstm_save', type=str,
                    metavar='PATH', help='model saved path')
parser.add_argument('--dataset-path', default='processOrder/datasets', type=str,
                    metavar='PATH', help='dataset path')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='BS',
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on all nodes when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--num_classes1', default=100, type=int,
                    metavar='N', help='the number of position labels')
parser.add_argument('--num_classes2', default=2, type=int,
                    metavar='N', help='the number of angle labels (latitude and longitude)')
parser.add_argument('--len', default=6, type=int,
                    metavar='LEN', help='the number of model input sequence length (containing the end point frame)')
parser.add_argument('--lr', default=0.001, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--wd', default=0.1, type=float,
                    metavar='WD', help='weight decay rate')
parser.add_argument('--T-0', default=5, type=int,
                    metavar='T0', help='T_0')
parser.add_argument('--T-mult', default=2, type=int,
                    metavar='TMULT', help='T_mult')

parser.add_argument('--pretrained', default='', type=str,
                    metavar='PATH', help='path to moco pretrained checkpoint')
parser.add_argument('--resume', default='', type=str,
                    metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate',
                    action='store_true', help='evaluate model on validation set')

parser.add_argument('-j', '--workers', default=0, type=int,
                    help='number of data loading workers (default: 32)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')


def main():
    """
    main program entry, responsible for multiprocessing distributed
    :return:
    """
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    """
    multiprocessing distributed process control: loading model, dataset, training and testing, saving checkpoint
    :param gpu: current gpu
    :param ngpus_per_node: the number of gpus of one machine
    :param args: program parameters
    :return:
    """
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass

        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        torch.distributed.barrier()

    # create model
    print("=> creating model")
    backbone = torchvision_models.mobilenet_v3_small(weights=(MobileNet_V3_Small_Weights.IMAGENET1K_V1))
    model = LSTM(
        backbone=backbone,
        extractor_dim=576,
        num_classes1=args.num_classes1,
        num_classes2=args.num_classes2,
        dim=512,
        num_layers=4,
        len=args.len
    )

    # mobilenetv3+lstm4: flops: 419.83 M, params: 10.65 M
    flops, params = profile(model,
                            (torch.randn((1, args.len, 3, 224, 224)),
                             torch.randn((1, args.len, 2))))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))

    # load from pre-trained, before DistributedDataParallel constructor
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")

            state_dict = checkpoint['state_dict']
            args.start_epoch = 0
            model.load_state_dict(state_dict, strict=False)

            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / args.world_size)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()

    # define loss function
    criterion1 = nn.CrossEntropyLoss().cuda(args.gpu)
    criterion2 = nn.MSELoss().cuda(args.gpu)
    criterion = UncertaintyLoss()

    # optimize model parameters and loss parameters
    parameters = list(filter(lambda p: p.requires_grad, model.parameters())) + \
                 list(filter(lambda p: p.requires_grad, criterion.parameters()))
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.wd)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.T_0, T_mult=args.T_mult)

    best_acc1 = 0
    best_acc2 = 0
    best_acc3 = math.inf

    # load from resume, start training from a certain epoch
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            best_acc2 = checkpoint['best_acc2']
            best_acc3 = checkpoint['best_acc3']
            print("best_acc1: " + str(best_acc1))
            print("best_acc2: " + str(best_acc2))
            print("best_acc3: " + str(best_acc3))

            model.load_state_dict(checkpoint['state_dict'], strict=False)
            criterion.load_state_dict(checkpoint['loss_weight'], strict=False)
            print(criterion.params)
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # reduce CPU usage, use it after the model is loaded onto the GPU
    torch.set_num_threads(1)
    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])
    # load dataset
    train_dataset = OrderTrainDataset(dataset_path=args.dataset_path, transform=train_transform, input_len=args.len - 1)
    test_dataset = OrderTestDataset(dataset_path=args.dataset_path, transform=val_transform, input_len=args.len - 1)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        loss, loss1, loss2, loss3 = train(train_loader, model, criterion1, criterion2, criterion, optimizer, lr_scheduler, epoch,
                                          args)

        # evaluate on validation set
        label_acc, target_acc, angle_acc_avg = validate(val_loader, model, args)

        # remember best acc@1 and save checkpoint
        label_is_best = label_acc >= best_acc1
        best_acc1 = max(label_acc, best_acc1)

        target_is_best = target_acc >= best_acc2
        best_acc2 = max(target_acc, best_acc2)

        angle_avg_is_best = angle_acc_avg <= best_acc3
        best_acc3 = min(angle_acc_avg, best_acc3)

        if not args.multiprocessing_distributed \
                or (args.multiprocessing_distributed and args.rank == 0):

            if os.path.exists(args.save_dir) is False:
                os.mkdir(args.save_dir)
            with open(args.save_dir + "/loss.txt", "a") as file1:
                file1.write(str(loss) + " " + str(loss1) + " " + str(loss2) + " " + str(loss3) + "\n")
            file1.close()
            with open(args.save_dir + "/label_acc.txt", "a") as file1:
                file1.write(str(label_acc) + " " + str(best_acc1) + "\n")
            file1.close()
            with open(args.save_dir + "/target_acc.txt", "a") as file1:
                file1.write(str(target_acc) + " " + str(best_acc2) + "\n")
            file1.close()
            with open(args.save_dir + "/angle_acc.txt", "a") as file1:
                file1.write(str(angle_acc_avg) + " " + str(best_acc3) + "\n")
            file1.close()

            for name, param in criterion.named_parameters():
                print(name)
                print(param)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'loss_weight': criterion.state_dict(),
                'best_acc1': best_acc1,
                'best_acc2': best_acc2,
                'best_acc3': best_acc3,
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
            }, label_is_best=label_is_best,
                target_is_best=target_is_best,
                angle_avg_is_best=angle_avg_is_best,
                args=args)


def train(train_loader, model, criterion1, criterion2, criterion, optimizer, lr_scheduler, epoch, args):
    """
    training process for one epoch
    :param train_loader: train dataloader
    :param model: baseline model (mobilenetv3+lstm)
    :param criterion1: ce
    :param criterion2: mse
    :param criterion: uncertainty loss
    :param optimizer: adamw
    :param lr_scheduler: CosineAnnealingWarmRestarts
    :param epoch: 120
    :param args:
    :return: total loss and three task losses
    """
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    label_top = AverageMeter('LabelAcc@1', ':6.2f')
    target_top = AverageMeter('TargetAcc@1', ':6.2f')
    angle_top = AverageMeter('AngleAcc@1', ':6.2f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, losses, label_top, target_top, angle_top],
        prefix="Epoch: [{}]".format(epoch))
    total_loss = 0
    total_loss1 = 0
    total_loss2 = 0
    total_loss3 = 0

    model.train()

    end = time.time()
    for i, (images, next_angles, label1, label2, label3) in enumerate(train_loader):
        if args.gpu is not None:
            # b,len,3,224,224
            images = images.cuda(args.gpu, non_blocking=True).to(dtype=torch.float32)
            # b,len,2
            next_angles = next_angles.cuda(args.gpu, non_blocking=True).to(dtype=torch.float32)
            # b
            label1 = label1.cuda(args.gpu, non_blocking=True).to(dtype=torch.int64)
            # b
            label2 = label2.cuda(args.gpu, non_blocking=True).to(dtype=torch.int64)
            # b,2
            label3 = label3.cuda(args.gpu, non_blocking=True).to(dtype=torch.float32)

        # b,len,3,224,224+b,len,2
        output1, output2, output3 = model(images, next_angles)

        loss1 = criterion1(output1, label1)
        loss2 = criterion1(output2, label2)
        loss3 = criterion2(output3, label3)

        loss = criterion([loss1, loss2, loss3])

        # measure accuracy and record loss
        label_acc, _ = accuracy(output1, label1, topk=(1, 5))
        target_acc, _ = accuracy(output2, label2, topk=(1, 5))
        angle_acc_avg = angle_diff(output3, label3)

        losses.update(loss.item(), images.size(0))
        label_top.update(label_acc[0], images.size(0))
        target_top.update(target_acc[0], images.size(0))
        angle_top.update(angle_acc_avg / images.size(0), images.size(0))

        total_loss += loss.item()
        total_loss1 += loss1.item()
        total_loss2 += loss2.item()
        total_loss3 += loss3.item()

        # compute gradient
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    return float(total_loss / (i + 1)), float(total_loss1 / (i + 1)), \
           float(total_loss2 / (i + 1)), float(total_loss3 / (i + 1))


def validate(val_loader, model, args):
    """
    validating process for one epoch
    :param val_loader: test dataloader
    :param model: baseline model (mobilenetv3+lstm)
    :param args:
    :return: three task accuracies
    """
    total_correct_label = 0
    total_correct_target = 0
    total_correct_angle_avg = 0
    total_samples = 0

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (images, next_angles, label1, label2, label3) in enumerate(val_loader):
            if args.gpu is not None:
                # b,len,3,224,224
                images = images.cuda(args.gpu, non_blocking=True).to(dtype=torch.float32)
                # b,len,2
                next_angles = next_angles.cuda(args.gpu, non_blocking=True).to(dtype=torch.float32)
                # b
                label1 = label1.cuda(args.gpu, non_blocking=True).to(dtype=torch.int64)
                # b
                label2 = label2.cuda(args.gpu, non_blocking=True).to(dtype=torch.int64)
                # b,2
                label3 = label3.cuda(args.gpu, non_blocking=True).to(dtype=torch.float32)

            # b,len,3,224,224+b,len,2
            output1, output2, output3 = model(images, next_angles)

            # measure accuracy
            _, preds = output1.max(1)
            num_correct = (preds == label1).sum()
            num_samples = preds.size(0)
            total_correct_label += num_correct.item()
            total_samples += num_samples

            _, preds = output2.max(1)
            num_correct = (preds == label2).sum()
            total_correct_target += num_correct.item()

            preds_avg = angle_diff(output3, label3)
            total_correct_angle_avg += preds_avg.item()

    label_acc = float(total_correct_label / total_samples)
    taget_acc = float(total_correct_target / total_samples)
    angle_acc_avg = float(total_correct_angle_avg / total_samples)

    print("Test: LabelAcc " + str(label_acc))
    print("Test: TargetAcc " + str(taget_acc))
    print("Test: AngleAccAvg " + str(angle_acc_avg))

    return label_acc, taget_acc, angle_acc_avg


def accuracy(output, target, topk=(1,)):
    """
    computes the accuracy over the k top predictions for the specified values of k
    :param output: actual output of the model
    :param target: ground truth label
    :param topk:
    :return: top-k acc
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def angle_diff(output, target):
    """
    compute Mean Absolute Angle Error between prediction and label
    :param output: actual output of the model
    :param target: ground truth label
    :return: Mean Absolute Angle Error within a batch
    """
    # b,2->b,1
    output_tan = output[:, 0] / output[:, 1]
    output_rad = torch.atan(output_tan)
    output_ang = output_rad * 180 / torch.pi

    # b,2->b,1
    target_tan = target[:, 0] / target[:, 1]
    target_rad = torch.atan(target_tan)
    target_ang = target_rad * 180 / torch.pi

    # since atan can only calculate [-90, 90], it needs to be converted to [-180, 180]
    for i in range(0, output.size(0)):
        if output[i, 0] >= 0 and output[i, 1] >= 0:
            continue
        elif output[i, 0] >= 0 and output[i, 1] <= 0:
            output_ang[i] += 180
        elif output[i, 0] <= 0 and output[i, 1] >= 0:
            continue
        elif output[i, 0] <= 0 and output[i, 1] <= 0:
            output_ang[i] -= 180

    for i in range(0, target.size(0)):
        if target[i, 0] >= 0 and target[i, 1] >= 0:
            continue
        elif target[i, 0] >= 0 and target[i, 1] <= 0:
            target_ang[i] += 180
        elif target[i, 0] <= 0 and target[i, 1] >= 0:
            continue
        elif target[i, 0] <= 0 and target[i, 1] <= 0:
            target_ang[i] -= 180

    diff = torch.abs(output_ang - target_ang)
    # deal with the situation when the gap is greater than 180
    diff = torch.where(diff > 180, 360 - diff, diff)

    return diff.sum()


def save_checkpoint(state, label_is_best, target_is_best, angle_avg_is_best, args):
    """
    save checkpoint
    :param state: model parameters
    :param label_is_best:
    :param target_is_best:
    :param angle_avg_is_best:
    :param args:
    :return:
    """
    torch.save(state, args.save_dir + "/checkpoint.pth.tar")
    if label_is_best:
        shutil.copyfile(args.save_dir + "/checkpoint.pth.tar",
                        args.save_dir + "/model_label_best.pth.tar")
    if target_is_best:
        shutil.copyfile(args.save_dir + "/checkpoint.pth.tar",
                        args.save_dir + "/model_target_best.pth.tar")
    if angle_avg_is_best:
        shutil.copyfile(args.save_dir + "/checkpoint.pth.tar",
                        args.save_dir + "/model_angle_avg_best.pth.tar")


class AverageMeter(object):
    """
    Computes and stores the average and current value
    """
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


if __name__ == '__main__':
    main()
