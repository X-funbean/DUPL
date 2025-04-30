import itertools
import logging
import os
import os.path as osp
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from tabulate import tabulate
from termcolor import colored
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm
from yacs.config import CfgNode as CN
from torch.utils.data.dataloader import DataLoader as TorchDataLoader
from dassl.data.data_manager import (
    DataManager,
    DatasetWrapper,
    build_sampler,
    build_dataset,
    build_transform,
)
from dassl.engine import TRAINER_REGISTRY, TrainerBase, TrainerX
from dassl.evaluation.evaluator import EvaluatorBase
from dassl.utils import load_checkpoint  # setup_logger
from libs.modeling.clip import clip
from libs.modeling.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from libs.trainers.pl_base_trainer import PlBaseTrainer, TextEncoder, load_clip_to_cpu
from libs.trainers.pldg_base_trainer import DGPlBaseTrainer

logger = logging.getLogger(
    f'fastdg.{os.path.relpath(__file__).replace(os.path.sep, ".")}'
)


def build_osdg_data_loader(
    cfg,
    dataset_wrapper,
    sampler_type="SequentialSampler",
    data_source=None,
    batch_size=64,
    n_domain=0,
    n_ins=2,
    is_train=True,
):
    # Build sampler
    sampler = build_sampler(
        sampler_type,
        cfg=cfg,
        data_source=data_source,
        batch_size=batch_size,
        n_domain=n_domain,
        n_ins=n_ins,
    )

    # Build data loader
    data_loader = TorchDataLoader(
        dataset_wrapper(cfg, data_source, transform=tfm, is_train=is_train),
        batch_size=batch_size,
        sampler=sampler,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        drop_last=is_train and len(data_source) >= batch_size,
        pin_memory=(torch.cuda.is_available() and cfg.USE_CUDA),
    )
    assert len(data_loader) > 0

    return data_loader


class OSDG_DatasetWrapper(DatasetWrapper):

    def __init__(self, cfg, data_source, ood_idx, transform=None, is_train=False):
        super().__init__(cfg, data_source, transform, is_train)
        self.ood_idx = ood_idx

    def __getitem__(self, idx):
        output = super().__getitem__(idx)
        item = self.data_source[idx]
        gt_label = item.label
        osdg_label = gt_label if gt_label < self.ood_idx else self.ood_idx
        output["osdg_label"] = osdg_label
        return output


class OSDG_DataManager(DataManager):

    OSDG_src_label_cfg = {
        "PACS": [[0, 1, 3], [0, 2, 4], [1, 2, 5]],
        "VLCS": [[0, 1], [1, 2], [2, 3]],
        "OfficeHome": [
            list(range(0, 15)) + list(range(21, 32)),
            list(range(0, 9)) + list(range(15, 21)) + list(range(32, 43)),
            list(range(0, 3)) + list(range(9, 21)) + list(range(43, 54)),
        ],
        "DigitsDG": [[0, 1, 2], [2, 3, 4], [4, 5, 6]],
        "miniDomainNet": [
            list(range(0, 20)) + list(range(40, 60)),
            list(range(0, 10)) + list(range(20, 40)) + list(range(80, 90)),
            list(range(10, 20)) + list(range(40, 50)) + list(range(60, 80)),
        ],
        "DomainNet": [
            list(range(0, 20)) + list(range(30, 60)) + list(range(70, 100)),
            list(range(10, 50)) + list(range(90, 130)),
            list(range(60, 80))
            + list(range(140, 165))
            + list(range(180, 195))
            + list(range(210, 230)),
            list(range(130, 140))
            + list(range(160, 185))
            + list(range(195, 220))
            + list(range(250, 270)),
            list(range(20, 40)) + list(range(220, 250)) + list(range(270, 300)),
        ],
    }

    OSDG_tgt_label_cfg = {
        "PACS": [0, 1, 2, 3, 4, 5, 6],
        "VLCS": [0, 1, 2, 3, 4],
        "OfficeHome": [0, 3, 4, 9, 10, 15, 16, 21, 22, 23, 32, 33, 34, 43, 44, 45]
        + list(range(54, 65)),
        "DigitsDG": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "miniDomainNet": list(range(0, 5))
        + list(range(8, 18))
        + list(range(25, 35))
        + list(range(43, 48))
        + list(range(75, 80))
        + list(range(83, 88))
        + list(range(90, 126)),
        "DomainNet": list(range(0, 10))
        + list(range(70, 80))
        + list(range(120, 130))
        + list(range(180, 190))
        + list(range(230, 240))
        + list(range(280, 290))
        + list(range(300, 345)),
    }

    def __init__(self, cfg, custom_tfm_train=None, custom_tfm_test=None):
        # Load dataset
        dataset_name = cfg.DATASET.NAME
        dataset = build_dataset(cfg)
        src_label_cfg = self.OSDG_src_label_cfg[dataset_name]
        tgt_label_cfg = self.OSDG_tgt_label_cfg[dataset_name]

        classnames = dataset.classnames

        # classnames_by_domain = defaultdict(list)
        # trainset_by_domain = defaultdict(list)

        # prepare source training data
        train_set = []
        data_by_domain = dataset.split_dataset_by_domain(dataset.train_x)
        for d, d_data in data_by_domain.items():
            domain_data_by_label = dataset.split_dataset_by_label(d_data)
            for i in src_label_cfg[d]:
                # classnames_by_domain[d].append(classnames[i])
                # trainset_by_domain[d].extend(domain_data_by_label[i])
                train_set.extend(domain_data_by_label[i])

        known_class_idx = list(set(itertools.chain(*src_label_cfg)))
        known_classnames = [classnames[i] for i in known_class_idx]
        # known_classes = ",".join(known_classnames)
        ood_idx = len(known_class_idx)

        # prepare target test data
        test_set = []
        domain_data_by_label = dataset.split_dataset_by_label(dataset.test)
        for i in tgt_label_cfg:
            test_set.extend(domain_data_by_label[i])

        # Build transform
        if custom_tfm_train is None:
            tfm_train = build_transform(cfg, is_train=True)
        else:
            logger.info("* Using custom transform for training")
            tfm_train = custom_tfm_train

        if custom_tfm_test is None:
            tfm_test = build_transform(cfg, is_train=False)
        else:
            logger.info("* Using custom transform for testing")
            tfm_test = custom_tfm_test

        # TODO:
        train_loader = build_osdg_data_loader(
            cfg,
            dataset_wrapper=OSDG_DatasetWrapper(
                cfg, train_set, ood_idx, tfm_train, is_train=True
            ),
            sampler_type=cfg.DATALOADER.TRAIN_X.SAMPLER,
            data_source=train_set,
            batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
            n_domain=cfg.DATALOADER.TRAIN_X.N_DOMAIN,
            n_ins=cfg.DATALOADER.TRAIN_X.N_INS,
            is_train=True,
        )

        # Build val_loader
        # val_loader = None
        # if dataset.val:
        #     val_loader = build_data_loader(
        #         cfg,
        #         sampler_type=cfg.DATALOADER.TEST.SAMPLER,
        #         data_source=dataset.val,
        #         batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
        #         tfm=tfm_test,
        #         is_train=False,
        #         dataset_wrapper=dataset_wrapper,
        #     )

        # Build test_loader
        test_loader = build_osdg_data_loader(
            cfg,
            dataset_wrapper=OSDG_DatasetWrapper(
                cfg, test_set, ood_idx, tfm_test, is_train=False
            ),
            sampler_type=cfg.DATALOADER.TEST.SAMPLER,
            data_source=test_set,
            batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
            is_train=False,
        )

        # Attributes
        self._num_classes = dataset.num_classes
        self._num_source_domains = len(cfg.DATASET.SOURCE_DOMAINS)
        self._lab2cname = dataset.lab2cname
        self.known_class_idx = known_class_idx
        self.known_classnames = known_classnames
        self.ood_idx = ood_idx

        # Dataset and data-loaders
        self.dataset = dataset
        self.train_loader = train_loader
        # self.train_loader_u = train_loader_u
        self.val_loader = val_loader
        self.test_loader = test_loader

        if cfg.VERBOSE:
            self.show_dataset_summary(cfg)


class OpenMax_Evaluator(EvaluatorBase):

    def __init__(self, cfg, lab2cname=None, tailsize=25, alpha=10, euclid_weight=1.0):
        super().__init__(cfg)
        self._lab2cname = lab2cname
        self._correct = 0
        self._total = 0
        self._per_class_res = None
        self._y_true = []
        self._y_pred = []
        if cfg.TEST.PER_CLASS_RESULT:
            assert lab2cname is not None
            self._per_class_res = defaultdict(list)


@TRAINER_REGISTRY.register()
class OSDGPlBaseTrainer(DGPlBaseTrainer):

    def __init__(self, cfg):
        TrainerBase.__init__(self)
        self.check_cfg(cfg)

        if torch.cuda.is_available() and cfg.USE_CUDA:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Save as attributes some frequently used variables
        self.start_epoch = self.epoch = 0
        self.max_epoch = cfg.OPTIM.MAX_EPOCH
        self.output_dir = cfg.OUTPUT_DIR

        self.cfg = cfg
        self.build_data_loader()
        self.build_model()
        self.evaluator = OSDG_Evaluator(cfg, lab2cname=self.lab2cname)
        self.best_result = -np.inf

    def build_data_loader(self):
        """Create essential data-related attributes.

        A re-implementation of this method must create the
        same attributes (self.dm is optional).
        """
        dm = OSDG_DataManager(self.cfg)

        self.train_loader_x = dm.train_loader_x
        # self.train_loader_u = dm.train_loader_u  # optional, can be None
        self.val_loader = dm.val_loader  # optional, can be None
        self.test_loader = dm.test_loader

        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname  # dict {label: classname}

        self.dm = dm

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


def set_osdgpl_config(cfg):
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
