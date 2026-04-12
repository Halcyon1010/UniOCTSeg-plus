import torch
import random
import numpy as np


class TwoStreamSampler_semi_DDP(torch.utils.data.Sampler):
    def __init__(self, 
                 dataset, 
                 batch_size, 
                 num_replicas=None, 
                 rank=None):
        
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()

        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank

        # 计算每个进程处理的数据子集
        self.stream1_indices = [i for i in range(len(dataset.data_pths)) if 'unlabeled' not in dataset.data_pths[i]]
        self.stream2_indices = [i for i in range(len(dataset.data_pths)) if 'unlabeled' in dataset.data_pths[i]]

        # 分割数据子集
        self.local_stream1_indices = self.stream1_indices[self.rank::self.num_replicas]
        self.local_stream2_indices = self.stream2_indices[self.rank::self.num_replicas]

    def __iter__(self):
        # 生成两个流的索引
        stream2_batch = [self.local_stream2_indices[i:i + self.batch_size] for i in
                         range(0, len(self.local_stream2_indices), self.batch_size)]

        stream1_batch = [random.choices(self.local_stream1_indices, k=self.batch_size) for _ in
                         range(len(stream2_batch))]

        # 交替返回两个流的索引
        combined_batches = [item for pair in zip(stream1_batch, stream2_batch) for item in pair]

        # 转换为全局索引
        global_indices = []
        for batch in combined_batches:
            for idx in batch:
                global_idx = self.local_stream1_indices[0] + idx - self.local_stream1_indices[0]
                if global_idx < len(self.dataset.data_pths):
                    global_indices.append(global_idx)
                else:
                    raise IndexError(
                        f"Global index {global_idx} is out of range for dataset of length {len(self.dataset.data_pths)}")

        return iter(global_indices)

    def set_epoch(self, epoch):
        self.epoch = epoch  # 更新epoch计数器

    def __len__(self):
        return len(self.local_stream1_indices) + len(self.local_stream2_indices)
    

class TwoStreamSampler_Semi(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size):
        super(TwoStreamSampler_Semi, self).__init__(dataset)
        self.dataset = dataset
        self.stream1_indices = [i for i in range(len(dataset.data_pths)) if 'unlabeled' not in dataset.data_pths[i]]
        self.stream2_indices = [i for i in range(len(dataset.data_pths)) if 'unlabeled' in dataset.data_pths[i]]
        self.batch_size = batch_size

    def __iter__(self):
        # 生成两个流的索引
        stream1_batch = [self.stream1_indices[i:i + self.batch_size] for i in
                         range(0, len(self.stream1_indices), self.batch_size)]
        stream2_batch = [random.choices(self.stream2_indices, k=self.batch_size) for _ in range(len(stream1_batch))]

        # 交替返回两个流的索引
        combined_batches = [item for pair in zip(stream1_batch, stream2_batch) for item in pair]
        return iter([item for batch in combined_batches for item in batch])

    def __len__(self):
        return len(self.stream1_indices) + len(self.stream2_indices)