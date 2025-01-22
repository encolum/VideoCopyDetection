import logging
import sys, os
import argparse
import random
import torch
import torch.nn.functional as F
import numpy as np
from mmcv import Config
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from model_factory.datasets.videolmdb_dataset import LabelVideoLmdbDataSet
from model_factory.backbones.swinv2 import SwinTransformerV2
from model_factory.recognizers.simple_selfsup_recognizer import SimpleContrastRecognizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--t', type=float, default=0.05)
    parser.add_argument('--margin', type=float, default=0.0)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--work_dir', type=str, default='')
    # parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--clip_grad_norm', type=float, default=1)
    # parser.add_argument('--local_rank', type=int, default=0)  
    # parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--instance_mask', action='store_true', default=False)
    parser.add_argument('--entropy_loss', action='store_true', default=False)
    parser.add_argument('--do_ema', action='store_true', default=False)
    parser.add_argument('--do_fgm', action='store_true', default=False)
    parser.add_argument('--entropy_weight', type=float, default=30)
    parser.add_argument('--ici_weight', type=float, default=1.)
    # parser.add_argument('--fp16', action='store_true', default=False)  
    parser.add_argument('--checkpointing', action='store_true', default=False)
    parser.add_argument('--concat_dataset', action='store_true', default=False)
    parser.add_argument('--product_loss', action='store_true', default=False)

    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Warning: Unrecognized arguments {unknown} will be ignored.")
    return args
args = parse_args()

work_dir =  r"D:\VideoMatching_latest\train\train_v115"
batch_size = args.batch_size
num_workers = args.num_workers
lr = args.lr
epochs = args.epochs
print_freq = args.print_freq
warmup_ratio = args.warmup_ratio

def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

setup_seed(args.seed)

os.makedirs(work_dir, exist_ok=True)
os.makedirs(f"{work_dir}/checkpoints", exist_ok=True) 
logger = logging.getLogger('log')
logger.setLevel(logging.INFO)

ch = logging.StreamHandler(stream=sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter("[%(levelname)s: %(asctime)s] %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

fh = logging.FileHandler(work_dir + '/log.txt')
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
logger.addHandler(fh)

cfg = Config.fromfile(r"D:\VideoMatching_latest\train\train_v115\config_v115.py")
cfg_data = cfg.data
cfg_model = cfg.model

train_dataset = LabelVideoLmdbDataSet(
        vids_path=cfg_data['vids_path'],
        meta_path=cfg_data['meta_path'],
        preprocess=cfg_data['preprocess'],
        lmdb_path=cfg_data['lmdb_path'],
        lmdb_size=cfg_data['lmdb_size'],
        width=cfg_data['width'],
        ann_path=cfg_data['ann_path'],
        arg_lmdb_path=cfg_data['arg_lmdb_path'],
        probs=cfg_data['probs'],
        crop=cfg_data['crop'],
        mixup=cfg_data['mixup'],)

train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
backbone_cfg = cfg_model["backbone"]
model = SimpleContrastRecognizer(backbone=backbone_cfg)
model.cuda()
opt = AdamW(model.parameters(), lr=lr)
batch_size = batch_size  
stepsize = (len(train_dataset) // batch_size + 1)
total_steps = (len(train_dataset) // batch_size + 1) * epochs
scheduler = get_linear_schedule_with_warmup(opt, num_warmup_steps=warmup_ratio * total_steps, num_training_steps=total_steps)

start_epoch = 0

def all_gather(local_rank, world_size, **tensors):
    tensors = list(tensors.values())
    _dims = [t.shape[-1] for t in tensors]
    tensors = torch.cat(tensors, dim=-1)
    tensors_all = [torch.zeros_like(tensors) for _ in range(world_size)]
    tensors_all[local_rank] = tensors
    tensors_all = torch.cat(tensors_all, dim=0)

    results = list()
    dimStart = 0
    assert sum(_dims) == tensors_all.shape[-1]
    for d in _dims:
        results.append(tensors_all[..., dimStart: dimStart + d])
        dimStart += d

    return tuple(results)

def contrast_loss_fn(emb_a, emb_b, temperature, margin=0):
    bz = emb_a.size(0)
    emb = torch.cat([emb_a, emb_b], dim=0)
    sims = emb @ emb.t()
    diag = torch.eye(sims.size(0), device=sims.device)
    
    small_value = torch.tensor(-10000., device=sims.device, dtype=sims.dtype)
    sims = torch.where(diag.eq(0), sims, small_value)
    # Ground truth for positive pairs
    gt = torch.cat([torch.arange(bz, device=sims.device) + bz, torch.arange(bz, device=sims.device)], dim=0)
    # Compute loss
    if margin > 0:
        loss_ = F.cross_entropy((sims - diag * margin) / temperature, gt)
    else:
        loss_ = F.cross_entropy(sims / temperature, gt)

    if loss_.dim() > 0:
        loss_ = loss_.mean() # thêm

    return loss_

def entropy_loss_fn(sims):
    device = sims.device
    diag = torch.eye(sims.size(0), device=device)
    local_mask = (1 - diag)
    small_value = torch.tensor(-10000., device=device, dtype=sims.dtype)
    # Mask out diagonal to find maximum non-matching similarity
    max_non_match_sim = torch.where(local_mask.bool(), sims, small_value).max(dim=1)[0]
    # Compute closest distance and apply log
    closest_distance = (1 / 2 - max_non_match_sim / 2).clamp(min=1e-6).sqrt()
    entropy_loss_ = -closest_distance.log().mean() * args.entropy_weight

    if entropy_loss_.dim() > 0:
        entropy_loss_ = entropy_loss_.mean()

    return entropy_loss_

def train_step(batch_data):
    # Move video data to GPU if it's not already there
    vid_a, vid_b = batch_data["vid_a"].cuda(), batch_data["vid_b"].cuda()
    bz = batch_data["img_a"].size(0)
    
    # Move image data to GPU
    device = batch_data["img_a"].device
    cat_x = torch.cat([batch_data["img_a"], batch_data["img_b"]], dim=0).to(device)

    # Process embeddings
    embeds = model(x=cat_x).to(device)  # Forward pass through the model
    embeds_norm = embeds / embeds.norm(dim=1, keepdim=True)  # Normalize embeddings
    emb_a, emb_b = embeds[:bz], embeds[bz:2 * bz]
    emb_a_norm, emb_b_norm = embeds_norm[:bz], embeds_norm[bz:2 * bz]

    # Compute similarity matrix
    sims_norm = emb_a_norm @ emb_b_norm.t()

    # Calculate loss
    if args.product_loss:
        match_sim = (emb_a_norm * emb_b_norm).sum(dim=1)
        entropy_loss_ = (1 - match_sim).exp().mean().to(device)
    else: 
        entropy_loss_ = entropy_loss_fn(sims_norm)

    # Contrastive loss
    ici_loss_ = contrast_loss_fn(emb_a_norm, emb_b_norm, args.t) * args.ici_weight

    return ici_loss_, entropy_loss_

global_step = 0
for _e in range(start_epoch, epochs):
    print(f"Epoch: {_e}")
    print('Training model')
    model.train()
    for _b, batch in enumerate(train_loader):
        print(f"Batch: {_b}")
        for _k, _v in batch.items():
            if isinstance(_v, torch.Tensor):
                batch[_k] = _v.cuda()
        print('Zero_grad')
        opt.zero_grad()
        print("Zero_grad done")
        print('ici + entropy loss')
        ici_loss, entropy_loss = train_step(batch)
        print('ici + entropy loss done')
        loss = ici_loss + entropy_loss
        print('backpropagating loss')
        loss.backward()
        print('backpropagation done')
        print('step')
        opt.step()
        scheduler.step()
        print('step done')

        global_step += 1
        if _b % print_freq == 0:
            logger.info('Epoch %d Batch %d Loss %.3f, ICI Loss %.3f, Entropy loss %.3f.' % (
                _e, _b, loss.item(), ici_loss.item(), entropy_loss.item())
            )

    ckpt = {'state_dict': model.state_dict(), 'optimizer': opt.state_dict(), 'scheduler': scheduler.state_dict(),
            'epoch': _e}
    print('Saving checkpoint')
    torch.save(ckpt, work_dir + '/checkpoints/epoch_%d.pth' % _e)

