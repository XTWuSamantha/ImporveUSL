# %%
import os
os.environ["USL_MODE"] = "FINETUNE"


# %%

from utils import cfg, logger, print_b, print_y
import utils
import models.resnet_cifar as resnet_cifar
import models.resnet_cifar_cld as resnet_cifar_cld
import models.resnet_medmnist as resnet_medmnist
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import medmnist
from medmnist import INFO, Evaluator
import random

# %%
utils.init(default_config_file="configs/bloodmnist_usl_finetune.yaml")
logger.info(cfg)

# %%
print_b("Loading dataset")
info = INFO[cfg.DATASET.NAME]
num_classes = len(info['label'])


selected_inds = np.load(cfg.FINETUNE.LABELED_INDICES_PATH)

if len(selected_inds) <= 40:
    logger.info(f"Labeled Indices: {repr(selected_inds)}")

train_dataset_cifar, val_dataset = utils.train_dataset_medmnist(
    transform_name=cfg.DATASET.TRANSFORM_NAME)



train_dataset_cifar.imgs = train_dataset_cifar.imgs[selected_inds]
select_targets = np.array(train_dataset_cifar.labels)[selected_inds]
target = []
for t in range(len(select_targets)):
    target = np.append(target,int(select_targets[t]))
targetnp = target.astype(int)
train_dataset_cifar.labels = list(targetnp)
assert len(train_dataset_cifar.imgs) == len(train_dataset_cifar.labels)

print(len(train_dataset_cifar.imgs))

# %%
print("Target dist:", np.unique(train_dataset_cifar.labels, return_counts=True))

if len(np.unique(train_dataset_cifar.labels)) != num_classes:
    logger.warning(f"WARNING: insufficient target: {len(np.unique(train_dataset_cifar.labels))} classes only")


if cfg.FINETUNE.REPEAT_DATA:
    train_dataset_cifar.imgs = np.vstack([train_dataset_cifar.imgs] * cfg.FINETUNE.REPEAT_DATA)
    train_dataset_cifar.labels = list(np.hstack([train_dataset_cifar.labels] * cfg.FINETUNE.REPEAT_DATA))

logger.info(f"Nunber of training samples: {len(train_dataset_cifar.labels)}")

train_dataloader = DataLoader(train_dataset_cifar, num_workers=cfg.DATALOADER.WORKERS,
                              batch_size=cfg.DATALOADER.BATCH_SIZE, pin_memory=True,
                              drop_last=True, shuffle=True)
val_dataloader = DataLoader(val_dataset, num_workers=cfg.DATALOADER.WORKERS,
                            batch_size=cfg.DATALOADER.BATCH_SIZE, pin_memory=True,
                            drop_last=False, shuffle=False)

train_targets = torch.tensor(train_dataset_cifar.labels)

val_targets = torch.tensor(val_dataset.labels)
train_targets.shape, val_targets.shape

print(train_targets.size())

# %%
print_b("Loading model")

checkpoint = torch.load(cfg.MODEL.PRETRAIN_PATH, map_location="cpu")

state_dict = utils.single_model(checkpoint["train_model"])

for k in list(state_dict.keys()):
    if k.startswith('linear') or k.startswith('fc') or k.startswith('groupDis'):
        del state_dict[k]

model = resnet_medmnist.__dict__[cfg.MODEL.ARCH]().cuda()

mismatch = model.load_state_dict(state_dict, strict=False)

logger.warning(
    f"Key mismatches: {mismatch} (extra contrastive keys are intended)")

# %%
print_b("Initializing optimizer")
torch.backends.cudnn.benchmark = True
if cfg.FINETUNE.FREEZE_BACKBONE:
    for name, param in model.named_parameters():
        if "linear" in name or "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

# params = [param for param in model.parameters() if param.requires_grad]
params = model.parameters()

optimizer = torch.optim.__dict__[cfg.OPTIMIZER.NAME](params,
                                                     lr=cfg.OPTIMIZER.LR,
                                                     momentum=cfg.OPTIMIZER.MOMENTUM,
                                                     nesterov=cfg.OPTIMIZER.NESTEROV,
                                                     weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY)

# %%
def train():
    model.train()

    for imgs, targets in tqdm(train_dataloader):
        imgs = imgs.cuda()
        targets = targets.cuda()

        pred = model(imgs)

        loss = F.cross_entropy(pred, targets)

        # print("Loss:", loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# %%
def eval():
    model.eval()

    preds = []
    all_targets = []

    confusion_matrix = torch.zeros((num_classes, num_classes))


    with torch.no_grad():
        for imgs, targets in tqdm(val_dataloader):
            imgs = imgs.cuda()

            pred = model(imgs)
            
            pred = pred.argmax(dim=1)

            targeti = []
            for t in range(len(targets)):
                targeti = np.append(targeti,targets[t])
            targetnpi = targeti.astype(int)

            targets = torch.tensor(targetnpi)

            for t, p in zip(targets.view(-1), pred.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

            preds.append(pred)
            all_targets.append(targets)

    preds = torch.cat(preds, dim=0).cpu()
    all_targets = torch.cat(all_targets, dim=0)

    acc = torch.mean((preds == all_targets).float()).item()

    acc_per_class = confusion_matrix.diag()/confusion_matrix.sum(1)

    print("Eval Acc:", acc)
    print("Acc per class:", acc_per_class)

    return acc

# import ipdb; ipdb.set_trace()

# %%
for epoch in range(cfg.EPOCHS):
    print("Epoch", epoch + 1)
    train()
    acc = eval()

# %%
print("Final Acc:", acc)
