import os
from datetime import datetime
import argparse
# import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.distributed as dist
from apex.parallel import DistributedDataParallel as DDP
from apex import amp

import time
from torch.profiler import profile, ProfilerActivity

intra_partitions = 2
batch_size = 20
batch_per_p = batch_size // intra_partitions
in_features = 1024
hidden = 1024
out_features = 1024
iteration = 10
total_step = iteration

params_dtype = torch.float32


def is_rank_0():
    # if torch.cuda.current_device() == 0:
    if torch.distributed.get_rank() == 0:
        return True
    else:
        return False


def get_stream(idx=0):
    if idx == 0:
        return _GLOBAL_STREAM0
    elif idx == 1:
        return _GLOBAL_STREAM1
    elif idx == 2:
        return _GLOBAL_STREAM2
    elif idx == 3:
        return _GLOBAL_STREAM3
    else:
        return None


def set_stream():
    global _GLOBAL_STREAM0
    global _GLOBAL_STREAM1
    global _GLOBAL_STREAM2
    global _GLOBAL_STREAM3
    # _GLOBAL_STREAM0 = torch.cuda.Stream()
    _GLOBAL_STREAM0 = torch.cuda.current_stream()
    _GLOBAL_STREAM1 = torch.cuda.Stream()
    _GLOBAL_STREAM2 = torch.cuda.Stream()
    _GLOBAL_STREAM3 = torch.cuda.Stream()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=2, type=int, metavar='N',
                        help='number of total epochs to run')
    args = parser.parse_args()
    args.world_size = args.gpus * args.nodes
    set_stream()

    train(args.local_rank, args)

    # os.environ['MASTER_ADDR'] = '10.57.23.164'
    os.environ['MASTER_PORT'] = '1357'
    # mp.spawn(train, nprocs=args.gpus, args=(args,))


class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

class FCNet(nn.Module):

    def __init__(self, in_features, hidden, out_features):
        super(FCNet, self).__init__()
        self.fc0 = torch.nn.Linear(in_features, hidden)
        self.fc1 = torch.nn.Linear(hidden, hidden)
        self.fc2 = torch.nn.Linear(hidden, out_features)

    def forward(self, x):
        x = self.fc0(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class FCNet1(nn.Module):

    def __init__(self, in_features, hidden, out_features):
        super(FCNet1, self).__init__()
        self.fc0 = BatchParallelLinear(in_features, hidden)
        self.fc1 = BatchParallelLinear(hidden, hidden)
        self.fc2 = BatchParallelLinear(hidden, out_features)

    def forward(self, x0, x1):
        x0, x1 = self.fc0(x0, x1)
        x0, x1 = self.fc1(x0, x1)
        x0, x1 = self.fc2(x0, x1)
        return x0, x1


class BatchParallelLinear(torch.nn.Module):

    def __init__(
        self,
        input_size,
        output_size,
        bias=True,
    ):
        super(BatchParallelLinear, self).__init__()
        self.weight = Parameter(
            torch.empty(output_size, input_size, dtype=params_dtype)
        )
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input0, input1):
        # stream0 = get_stream(0)
        # with torch.cuda.stream(stream0):
        output0 = torch.matmul(input0, self.weight.t())
    
        # stream1 = get_stream(1)
        # with torch.cuda.stream(stream1):
        output1 = torch.matmul(input1, self.weight.t())
        return output0, output1


def loss_func1(criterions, outputs, labels):
    criterion0, criterion1 = criterions[0], criterions[1]
    out0, out1 = outputs[0], outputs[1]
    label0, label1 = labels[0], labels[1]

    # stream0 = get_stream(0)
    # with torch.cuda.stream(stream0):
    loss0 = criterion0(out0, label0)

    # stream1 = get_stream(1)
    # with torch.cuda.stream(stream1):
    loss1 = criterion1(out1, label1)
    
    loss = loss0 + loss1
    return loss 

def train(gpu, args):
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
    torch.manual_seed(0)
    # model = ConvNet()
    model = FCNet1(in_features, hidden, out_features)
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    criterion1 = nn.CrossEntropyLoss().cuda(gpu)

    optimizer = torch.optim.SGD(model.parameters(), 1e-4)
    # Wrap the model
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    # Data loading code
    # train_dataset = torchvision.datasets.MNIST(root='./data',
    #                                            train=True,
    #                                            transform=transforms.ToTensor(),
    #                                            download=True)
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
    #                                                                 num_replicas=args.world_size,
    #                                                                 rank=rank)
    # train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
    #                                            batch_size=batch_size,
    #                                            shuffle=False,
    #                                            num_workers=0,
    #                                            pin_memory=True,
    #                                            sampler=train_sampler)

    # start = datetime.now()
    # # total_step = len(train_loader)
    # for epoch in range(args.epochs):
    #     # for i, (images, labels) in enumerate(train_loader):

    #     current_time = time.time()
    #     local_time = time.localtime(current_time)
    #     time_stamp = f'{local_time.tm_hour}-{local_time.tm_min}'
    #     def trace_handler(prof):
    #         # output = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
    #         # print(output)
    #         if is_rank_0():
    #             prof.export_chrome_trace(f"./traces/all-split1{time_stamp}.json")

    #     with profile(
    #         activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #         record_shapes=True,
    #         schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
    #         on_trace_ready=trace_handler
    #     ) as prof:
    #     # if True:
    #         for i in range(iteration):
    #             images = torch.randint(0, 128, (batch_size, in_features), dtype=torch.float32, device=gpu)
    #             labels = torch.randint(0, out_features, (batch_size,), dtype=torch.int64, device=gpu)

    #             # images0 = torch.randint(0, 128, (batch_per_p, in_features), dtype=torch.float32, device=gpu)
    #             # labels0 = torch.randint(0, out_features, (batch_per_p,), dtype=torch.int64, device=gpu)
    #             # images1 = torch.randint(0, 128, (batch_per_p, in_features), dtype=torch.float32, device=gpu)
    #             # labels1 = torch.randint(0, out_features, (batch_per_p,), dtype=torch.int64, device=gpu)
    #             images0 = torch.narrow(images, 0, 0*batch_per_p, batch_per_p)
    #             labels0 = torch.narrow(labels, 0, 0*batch_per_p, batch_per_p)
    #             images1 = torch.narrow(images, 0, 1*batch_per_p, batch_per_p)
    #             labels1 = torch.narrow(labels, 0, 1*batch_per_p, batch_per_p)

    #             # Forward pass
    #             output0, output1 = model(images0, images1)
    #             loss = loss_func1([criterion, criterion1], [output0, output1], [labels0, labels1])

    #             # images = images.cuda(non_blocking=True)
    #             # labels = labels.cuda(non_blocking=True)
    #             # Forward pass
    #             # outputs = model(images)
    #             # loss = criterion(outputs, labels)

    #             # Backward and optimize
    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()

    #             prof.step()

    #             # if (i + 1) % 100 == 0 and gpu == 0:
    #             if gpu == 0:
    #                 print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, args.epochs, i + 1, total_step,
    #                                                                         loss.item()))
    # if gpu == 0:
    #     print("Training complete in: " + str(datetime.now() - start))

    static_input0 = torch.randint(0, 128, (batch_per_p, in_features), dtype=torch.float32, device=gpu)
    static_labels0 = torch.randint(0, out_features, (batch_per_p,), dtype=torch.int64, device=gpu)
    static_input1 = torch.randint(0, 128, (batch_per_p, in_features), dtype=torch.float32, device=gpu)
    static_labels1 = torch.randint(0, out_features, (batch_per_p,), dtype=torch.int64, device=gpu)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(24):
            optimizer.zero_grad(set_to_none=True)
            static_y0, static_y1 = model(static_input0, static_input1)
            static_loss = loss_func1([criterion, criterion1], [static_y0, static_y1], [static_labels0, static_labels1])
            static_loss.backward()
            optimizer.step()
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.graph(g):
        static_y0, static_y1 = model(static_input0, static_input1)
        static_loss = loss_func1([criterion, criterion1], [static_y0, static_y1], [static_labels0, static_labels1])
        static_loss.backward()
        optimizer.step()

    # for i in range(iteration):
    #     images = torch.randint(0, 128, (batch_size, in_features), dtype=torch.float32, device=gpu)
    #     labels = torch.randint(0, out_features, (batch_size,), dtype=torch.int64, device=gpu)

    #     images0 = torch.narrow(images, 0, 0*batch_per_p, batch_per_p)
    #     labels0 = torch.narrow(labels, 0, 0*batch_per_p, batch_per_p)
    #     images1 = torch.narrow(images, 0, 1*batch_per_p, batch_per_p)
    #     labels1 = torch.narrow(labels, 0, 1*batch_per_p, batch_per_p)

    #     static_input1.copy_(images0)
    #     static_labels1.copy_(labels0)
    #     static_input.copy_(images1)
    #     static_target.copy_(labels1)
    #     # replay() includes forward, backward, and step.
    #     # You don't even need to call optimizer.zero_grad() between iterations
    #     # because the captured backward refills static .grad tensors in place.
    #     g.replay()


if __name__ == '__main__':
    main()