# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import logging

from .lr_scheduler import WarmupMultiStepLR


def make_optimizer(cfg, model):
    params = []
    logger = logging.getLogger(__name__)

    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue

        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS

        if 'LR_FACTOR_LAYER3' in cfg.SOLVER and 'backbone.body.layer3' in key and cfg.SOLVER.LR_FACTOR_LAYER3 > 0:
            lr *= cfg.SOLVER.LR_FACTOR_LAYER3
            logger.info("-> Setting LR for layer {} to {}".format(key, lr))
        elif 'LR_FACTOR_LAYER4' in cfg.SOLVER and 'backbone.body' in key and cfg.SOLVER.LR_FACTOR_LAYER4 > 0:
            lr *= cfg.SOLVER.LR_FACTOR_LAYER4
            logger.info("-> Setting LR for layer {} to {}".format(key, lr))
        elif 'LR_FACTOR_FPN_RPN' in cfg.SOLVER and cfg.SOLVER.LR_FACTOR_FPN_RPN > 0 and ('backbone.fpn' in key or 'rpn.head.conv' in key or 'rpn.head.cls_logits' in key):
            lr *= cfg.SOLVER.LR_FACTOR_FPN_RPN
            logger.info("-> Setting LR for layer {} to {}".format(key, lr))

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    optimizer = torch.optim.SGD(params, lr, momentum=cfg.SOLVER.MOMENTUM)
    return optimizer


def make_lr_scheduler(cfg, optimizer):
    return WarmupMultiStepLR(
        optimizer,
        cfg.SOLVER.STEPS,
        cfg.SOLVER.GAMMA,
        warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
        warmup_iters=cfg.SOLVER.WARMUP_ITERS,
        warmup_method=cfg.SOLVER.WARMUP_METHOD,
    )
