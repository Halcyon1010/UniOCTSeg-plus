import os
import random
import warnings
import argparse
from collections import OrderedDict

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from loss import (
    MultiSoftDiceLoss_v2,
    Multi_Cross_entropy,
    Bidirection_consistency_loss,
    dice_score,
    get_non_zero_target_v3,
    mask2onehot,
)
from OCTdataset.dataset import UniversalOCTTrainDataset, UniversalOCTDataset
from models.UniOCTSeg_Plus import UniOCTSeg_Plus
from models.utils import result_logger

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
warnings.filterwarnings("ignore")


def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_pth', type=str, default='/data/zhongj/data/',
                        help='Root directory of dataset txt files.')
    parser.add_argument('--result_path', type=str, default='/data/zhongj/Codes/UniOCT_pro/output_')
    parser.add_argument('--exp', type=str,
                        default='All_dataset_UniOCTSeg_plus',
                        help='Experiment name.')
    parser.add_argument('--resume', type=str,
                        default='',
                        help='Path to checkpoint.')
    parser.add_argument('--transformer_weights', type=str,
                        default='/data/zhongj/Codes/weights/imagenet21k_ViT-B_16.npz')

    parser.add_argument('--gen_Task', type=bool, default=True)
    parser.add_argument('--num_classes', type=int, default=2,
                        help='Output channel of network.')
    parser.add_argument('--max_iterations', type=int, default=80000,
                        help='Maximum training iterations.')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size per GPU.')
    parser.add_argument('--base_lr', type=float, default=3e-4,
                        help='Learning rate.')
    parser.add_argument('--patch_size', default=[256, 256],
                        help='Patch size of network input.')
    parser.add_argument('--num_workers', type=int, default=18,
                        help='Number of dataloader workers.')

    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed.')
    parser.add_argument('--deterministic', type=int, default=1,
                        help='Whether to use deterministic training.')
    parser.add_argument('--device', type=int, default=7,
                        help='Fallback single GPU id.')
    parser.add_argument('--local-rank', type=int, default=-1,
                        help='Local rank for distributed training.')

    return parser


class EMA:
    def __init__(self, decay=0.99):
        self.decay = decay
        self.ema_dict = OrderedDict()

    @staticmethod
    def _clean_name(name):
        return name.replace('module.', '')

    def copy_params(self, ema_model, model):
        self.ema_dict = OrderedDict()

        for name, param in model.named_parameters():
            self.ema_dict[self._clean_name(name)] = param.data.clone()

        for name, buffer in model.named_buffers():
            self.ema_dict[self._clean_name(name)] = buffer.data.clone()

        ema_model.load_state_dict(self.ema_dict)

    def update(self, ema_model, model):
        model.eval()

        for name, param in model.named_parameters():
            if param.requires_grad:
                clean_name = self._clean_name(name)
                new_average = (1 - self.decay) * param.data + self.decay * self.ema_dict[clean_name]
                self.ema_dict[clean_name] = new_average

        for name, buffer in model.named_buffers():
            clean_name = self._clean_name(name)
            new_average = (1 - self.decay) * buffer.data + self.decay * self.ema_dict[clean_name]
            self.ema_dict[clean_name] = new_average

        ema_model.load_state_dict(self.ema_dict)
        model.train()

    def resume(self, ema_model):
        self.ema_dict = OrderedDict()

        for name, param in ema_model.named_parameters():
            self.ema_dict[self._clean_name(name)] = param.data.clone()

        for name, buffer in ema_model.named_buffers():
            self.ema_dict[self._clean_name(name)] = buffer.data.clone()


def set_random_seed(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
    else:
        cudnn.benchmark = True
        cudnn.deterministic = False


def create_model(transformer_weights, task_num=9):
    return UniOCTSeg_Plus(task_num=task_num, transformer_weights=transformer_weights)


def build_dataloaders(args):
    train_dataset = UniversalOCTTrainDataset(
        root=os.path.join(args.data_pth, 'new_train_data.txt'),
        train_test_flag='train',
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=8,
        drop_last=False,
        shuffle=False,
        pin_memory=False
    )

    val_dataset = UniversalOCTDataset(
        root=os.path.join(args.data_pth, 'new_val_data.txt'),
        train_test_flag='val'
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=16,
        num_workers=8,
        drop_last=False,
        shuffle=False,
        pin_memory=False
    )

    return train_loader, val_loader


def load_checkpoint_if_needed(args, model, ema_model, optimizer, ema_helper):
    iter_num = 0
    best_performance = 0.0

    if not args.resume:
        return iter_num, best_performance

    if not os.path.exists(args.resume):
        return iter_num, best_performance

    checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model'])

    if 'ema_model' in checkpoint:
        ema_model.load_state_dict(checkpoint['ema_model'])
        best_performance = 0.0
    else:
        new_weights = OrderedDict()
        for k, v in checkpoint['model'].items():
            new_weights[k.replace('module.', '')] = v
        ema_model.load_state_dict(new_weights)
        best_performance = 0.0

    optimizer.load_state_dict(checkpoint['optimizer'])
    ema_helper.resume(ema_model)
    ema_helper.copy_params(ema_model, model)
    iter_num = checkpoint['iter_num'] + 1

    return iter_num, best_performance


def save_checkpoint(save_path, model, optimizer, iter_num, ema_model=None, layer_performance=None):
    save_dict = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iter_num': iter_num
    }

    if ema_model is not None:
        save_dict['ema_model'] = ema_model.state_dict()

    if layer_performance is not None:
        save_dict['layer_performance'] = layer_performance

    torch.save(save_dict, save_path)


def adjust_learning_rate(optimizer, base_lr, iter_num, max_iterations):
    lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_
    return lr_


def valid(model, val_loader, device, logger):
    model.eval()
    dice_list = []
    dataset_result = {}

    for imgs, onehot_labels, task_onehots, dataset_names, pth in val_loader:
        imgs = torch.as_tensor(imgs, device=device, dtype=torch.float32).unsqueeze(1)
        labels = torch.as_tensor(onehot_labels, device=device, dtype=torch.float32)
        task_onehots = torch.as_tensor(task_onehots, device=device, dtype=torch.long)

        with torch.no_grad():
            pred_logits = model(imgs, task_onehots)
            probs_n = pred_logits.contiguous()

        batch_size = probs_n.shape[0]

        for b in range(batch_size):
            probs, targets = get_non_zero_target_v3(
                probs_n[b:b + 1],
                labels[b:b + 1]
            )

            targets = torch.stack([
                torch.as_tensor(
                    mask2onehot(targets[0:1, c].cpu().numpy(), [0, 1]),
                    device=probs.device
                ).permute(1, 0, 2, 3)
                for c in range(targets.shape[1])
            ], dim=1)

            total_dice = [
                dice_score(
                    torch.softmax(probs[:, c], dim=1)[:, 1:2],
                    targets[:, c][:, 1:2]
                )
                for c in range(targets.shape[1])
            ]

            if dataset_names[b] not in dataset_result:
                dataset_result[dataset_names[b]] = []
            dataset_result[dataset_names[b]].append(total_dice)

    model.train()

    for dataset_name in dataset_result.keys():
        merged_result = np.concatenate(dataset_result[dataset_name], axis=0)
        mean_result = np.mean(dataset_result[dataset_name])
        std_result = np.mean(np.std(merged_result, axis=0))
        class_mean_result = np.mean(merged_result, axis=0)

        logger.write_content(dataset_name + ' : ', mean_result, '±', std_result)
        logger.write_content(dataset_name + ' : ', class_mean_result)

        dice_list.append(mean_result)

    return np.mean(dice_list)


def train(args, snapshot_path):
    logger = result_logger(snapshot_path)
    logger.write_content(str(args))

    device = torch.device("cuda", args.local_rank)

    model = create_model(args.transformer_weights, task_num=9)
    model = model.to(device)
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        broadcast_buffers=False,
        find_unused_parameters=True
    )

    ema_model = create_model(args.transformer_weights, task_num=9).to(device)
    ema_helper = EMA()

    train_loader, val_loader = build_dataloaders(args)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.base_lr,
        betas=(0.9, 0.95),
        weight_decay=0.0001
    )

    loss_dice = MultiSoftDiceLoss_v2()
    loss_ce = Multi_Cross_entropy()
    loss_consistency = Bidirection_consistency_loss(logger)

    logger.write_content(f"{len(train_loader)} iterations per epoch")

    iter_num, best_performance = load_checkpoint_if_needed(
        args, model, ema_model, optimizer, ema_helper
    )

    max_epoch = args.max_iterations // len(train_loader) + 1
    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch in iterator:
        if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        for idx, (imgs, labels, task_onehots, dataset_names) in enumerate(train_loader):
            imgs = torch.as_tensor(imgs, device=device, dtype=torch.float32).unsqueeze(1)
            labels = torch.as_tensor(labels, device=device, dtype=torch.float32)
            task_onehots = torch.as_tensor(task_onehots, device=device, dtype=torch.long)

            pred_logits = model(imgs, task_onehots)

            layer_loss_dice = loss_dice(pred_logits, labels)
            layer_loss_ce = loss_ce(pred_logits, labels)

            if iter_num > 1000:
                ema_helper.update(ema_model, model)

                with torch.no_grad():
                    ema_logits = ema_model(imgs, task_onehots)
                    ema_probs = torch.softmax(ema_logits, dim=2)[:, :, 1]

                probs = torch.softmax(pred_logits, dim=2)[:, :, 1]
                consistency_loss = loss_consistency(ema_probs, probs, task_onehots)

                total_loss = layer_loss_dice + layer_loss_ce + consistency_loss

                logger.write_content(
                    'iteration %d : loss : %.5f, loss_layer_dice: %.5f, loss_layer_ce: %.5f, loss_consis: %.5f'
                    % (
                        iter_num,
                        total_loss.item(),
                        layer_loss_dice.item(),
                        layer_loss_ce.item(),
                        consistency_loss.item()
                    )
                )
            else:
                total_loss = layer_loss_dice + layer_loss_ce

                logger.write_content(
                    'iteration %d : loss : %.5f, loss_layer_dice: %.5f, loss_layer_ce: %.5f'
                    % (
                        iter_num,
                        total_loss.item(),
                        layer_loss_dice.item(),
                        layer_loss_ce.item()
                    )
                )

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            adjust_learning_rate(optimizer, args.base_lr, iter_num, args.max_iterations)
            iter_num += 1

            if iter_num == 1000:
                ema_helper.copy_params(ema_model, model)
                save_path = os.path.join(snapshot_path, f'iter_{iter_num}.pth')
                save_checkpoint(save_path, model, optimizer, iter_num)
                logger.write_content(f"save model to {save_path}")

            if iter_num >= 20000 and iter_num % 1000 == 0 and dist.get_rank() == 0:
                model.eval()
                ema_dice = valid(ema_model, val_loader, device, logger)

                if best_performance <= ema_dice:
                    best_performance = ema_dice
                    logger.write_content(f"current best performance: {ema_dice:.6f}")
                    logger.write_content('task have been called:')

                    save_path = os.path.join(snapshot_path, f'ema_iter_{iter_num}.pth')
                    save_checkpoint(
                        save_path,
                        model,
                        optimizer,
                        iter_num,
                        ema_model=ema_model,
                        layer_performance=ema_dice
                    )
                    logger.write_content(f"save model to {save_path}")

                model.train()

            if iter_num >= args.max_iterations:
                break

        if iter_num >= args.max_iterations:
            iterator.close()
            break

    return "Training Finished!"


def main():
    parser = build_parser()
    args = parser.parse_args()

    set_random_seed(args.seed, deterministic=bool(args.deterministic))

    args.snapshot_path = args.result_path + args.exp
    os.makedirs(args.snapshot_path, exist_ok=True)

    dist.init_process_group(backend='nccl')
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)

    args.local_rank = local_rank
    args.world_size = dist.get_world_size()

    train(args, args.snapshot_path)


if __name__ == "__main__":
    main()
