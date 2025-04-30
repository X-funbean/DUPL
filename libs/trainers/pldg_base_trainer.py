import logging
import os
import os.path as osp

import torch
import torch.nn as nn
from tqdm import tqdm
from yacs.config import CfgNode as CN

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_checkpoint  # setup_logger,,
from libs.modeling.clip import clip
from libs.modeling.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from libs.trainers.pl_base_trainer import (PlBaseTrainer, TextEncoder,
                                           load_clip_to_cpu)

logger = logging.getLogger(
    f'fastdg.{os.path.relpath(__file__).replace(os.path.sep, ".")}'
)


@TRAINER_REGISTRY.register()
class DGPlBaseTrainer(PlBaseTrainer):

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        domain = batch["domain"]
        input = input.to(self.device, non_blocking=True)
        label = label.to(self.device, non_blocking=True)
        domain = domain.to(self.device, non_blocking=True)
        return input, label, domain

    def parse_batch_test(self, batch):
        input = batch["img"]
        label = batch["label"]
        domain = batch["domain"]
        input = input.to(self.device, non_blocking=True)
        label = label.to(self.device, non_blocking=True)
        domain = domain.to(self.device, non_blocking=True)
        return input, label, domain

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        logger.info(f"Evaluate on the *{split}* set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            images, labels, domains = self.parse_batch_test(batch)
            output = self.model_inference(images)
            self.evaluator.process(output, labels)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]


def set_dgpl_config(cfg):
    _C = cfg
    _C.DATASET.ROOT = "/root/xfb/datasets/DG"
    _C.DATASET.NAME = "PACS"
    _C.DATASET.SOURCE_DOMAINS = ("cartoon", "photo", "sketch")
    _C.DATASET.TARGET_DOMAINS = ("art_painting",)

    _C.TRAIN.VAL_FREQ = 0

    _C.TEST.FINAL_MODEL = "best_val"
    _C.TEST.PER_TARGET_RESULT = False

    # -----------------------------------------------------------------------------
    # DG
    # -----------------------------------------------------------------------------
    _C.DOMAINBED = CN()
    # _C.DOMAINBED.USE_FIXED_SPLIT = False
    _C.DOMAINBED.USE_FIXED_SPLIT = True
    _C.DOMAINBED.HOLDOUT_FRACTION = 0.2

    # _C.DOMAINBED.TRAIN_ITERS = 15000
    # _C.DOMAINBED.CHECKPOINT_PERIOD = 1000 # iters
    # _C.DOMAINBED.USE_DATASET_CONFIG = True
    # _C.DOMAINBED.RESNET_DROPOUT = 0. # [0., 0.1, 0.5]
    # _C.DOMAINBED.FREEZE_BN = True
