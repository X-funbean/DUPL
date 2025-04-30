import datetime
import itertools
import logging
import math
import os
import os.path as osp
import random
import sys
import time
from collections import OrderedDict, defaultdict
from copy import deepcopy

import IPython
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torchsnooper
import torchvision.transforms as T
from sklearn.metrics import confusion_matrix, f1_score
from tabulate import tabulate
from termcolor import colored
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data.dataloader import DataLoader as TorchDataLoader
from tqdm import tqdm
from yacs.config import CfgNode as CN

sys.path.append(".")

from dassl.config import clean_cfg
from dassl.data import DataManager
from dassl.data.data_manager import DatasetWrapper, build_data_loader, build_transform
from dassl.data.datasets import build_dataset
from dassl.data.datasets.base_dataset import Datum
from dassl.data.samplers import build_sampler
from dassl.engine import TRAINER_REGISTRY, TrainerBase, TrainerX
from dassl.evaluation import build_evaluator
from dassl.evaluation.evaluator import Classification, EvaluatorBase
from dassl.metrics import compute_accuracy
from dassl.optim import build_lr_scheduler, build_optimizer
from dassl.utils import load_checkpoint  # setup_logger
from dassl.utils import AverageMeter, MetricMeter, load_pretrained_weights, read_image
from libs.engine import (
    default_argument_parser,
    default_setup,
    get_cfg,
    launch,
    merge_from_args,
)
from libs.modeling.clip import clip
from libs.modeling.clip.model import build_model
from libs.modeling.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from libs.trainers.pl_base_trainer import PlBaseTrainer, TextEncoder, load_clip_to_cpu
from libs.trainers.pldg_base_trainer import (
    DGPlBaseTrainer,
    TextEncoder,
    load_clip_to_cpu,
    set_dgpl_config,
)
from libs.utils import comm

logger = logging.getLogger(
    f'fastdg.{os.path.relpath(__file__).replace(os.path.sep, ".")}'
)
_tokenizer = _Tokenizer()


class PromptLearner(nn.Module):

    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.METHOD.N_CTX
        n_unknown_token = cfg.METHOD.N_NEG
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert (
            cfg_imsize == clip_imsize
        ), f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        logger.info("Initializing a generic context")
        ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
        unknown_token = torch.empty(n_unknown_token, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        nn.init.normal_(unknown_token, std=0.02)
        prompt_prefix = " ".join(["X"] * n_ctx)

        logger.info(f'Initial context: "{prompt_prefix}"')
        logger.info(f"Number of context words (tokens): {n_ctx}")

        self.ctx = Parameter(ctx_vectors)  # to be optimized
        self.unknown_token = Parameter(unknown_token)

        classnames = [name.replace("_", " ") for name in classnames]
        unknown_placeholder = " ".join(["U"] * n_unknown_token)
        classnames.append(unknown_placeholder)
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        # prompts = [f"a {prompt_prefix} style of a {name}." for name in classnames]
        prompts = [f"{prompt_prefix} {name}." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        # IPython.embed()
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.n_unknown_token = n_unknown_token
        self.dtype = dtype
        self.ctx_dim = ctx_dim
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

    def forward(self):
        ctx = self.ctx

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls + 1, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,  # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )  # [C, 77, dim]
        prompts[-1, 1 + self.n_ctx : 1 + self.n_ctx + self.n_unknown_token] = (
            self.unknown_token
        )

        return prompts


class CustomCLIP(nn.Module):

    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        self.clip_model = clip_model
        self.text_encoder = TextEncoder(clip_model)
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts

        self.n_cls = self.prompt_learner.n_cls
        self.n_sample = cfg.METHOD.N_SAMPLE
        self.n_select = cfg.METHOD.N_SELECT
        self.n_sample_from = cfg.METHOD.N_SAMPLE_FROM

    @property
    def device(self):
        return next(self.clip_model.parameters()).device

    # @torchsnooper.snoop()
    def forward(self, images, osdg_labels=None, images_aug=None):
        img_feats = self.image_encoder(images.type(self.dtype))
        img_feats = F.normalize(img_feats, dim=-1)

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts

        txt_feats = self.text_encoder(prompts, tokenized_prompts)
        txt_feats = F.normalize(txt_feats, dim=-1)  # [C_k+1, d]

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * img_feats @ txt_feats.t()  # [B, C_k+1]

        if osdg_labels is not None:
            assert images_aug is not None
            img_aug_feats = self.image_encoder(images_aug.type(self.dtype))
            img_aug_feats = F.normalize(img_aug_feats, dim=-1)
            logits_aug = logit_scale * img_aug_feats @ txt_feats.t()  # [B, C_k+1]

            loss_dict = {}
            loss_dict["loss_ce"] = F.cross_entropy(logits, osdg_labels)
            loss_dict["loss_ua"] = self.cal_ua_loss(logits, osdg_labels)
            loss_dict["loss_ce_aug"] = F.cross_entropy(logits_aug, osdg_labels)
            loss_dict["loss_ua_aug"] = self.cal_ua_loss(logits_aug, osdg_labels)

            # loss_dict["loss_re1"] = F.mse_loss(logits.detach(), logits_aug)
            # loss_dict["loss_re2"] = F.mse_loss(F.normalize(logits[:, :-1], dim=-1).detach(),
            #                                    F.normalize(logits_aug[:, :-1], dim=-1)) * 0.1
            # loss_dict["loss_re3"] = (2 - F.cosine_similarity(logits.detach().float(), logits_aug.float())).mean()
            mask = logits.argmax(dim=-1) == osdg_labels
            if any(mask):
                loss_dict["loss_re4"] = (
                    F.mse_loss(logits[mask].detach(), logits_aug[mask]) * 0.01
                )
            # IPython.embed()

            return loss_dict, logits
        else:
            return logits

    def cal_ua_loss(self, logits, labels):
        B, C = logits.shape
        mask = torch.ones_like(logits).scatter_(1, labels.unsqueeze(1), 0.0).bool()
        filtered_logits = logits[mask].view(B, -1)  # [B, C_k]
        unknown_labels = torch.ones(B).to(labels) * (C - 2)
        loss = F.cross_entropy(filtered_logits, unknown_labels)
        return loss


def build_osdg_data_loader(
    cfg,
    dataset,
    data_source,
    ood_idx,
    batch_size=64,
    n_domain=0,
    n_ins=2,
    tfm=None,
    is_train=True,
    sampler_type="SequentialSampler",
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
        dataset=OSDG_DatasetWrapper(cfg, data_source, ood_idx, tfm, is_train, dataset),
        batch_size=batch_size,
        sampler=sampler,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        drop_last=is_train and len(data_source) >= batch_size,
        pin_memory=(torch.cuda.is_available() and cfg.USE_CUDA),
    )
    assert len(data_loader) > 0

    return data_loader


class OSDG_DatasetWrapper(DatasetWrapper):

    def __init__(
        self, cfg, data_source, ood_idx, transform=None, is_train=False, dataset=None
    ):
        super().__init__(cfg, data_source, transform, is_train)
        self.ood_idx = ood_idx
        if is_train:
            assert dataset is not None
            assert self.transform is not None
            self.data_by_domain = dataset.split_dataset_by_domain(data_source)
            self.pre_tfms = T.Compose(
                self.transform.transforms[:-2] + [lambda x: np.asarray(x)]
            )
            self.post_tfms = T.Compose(self.transform.transforms[-2:])

    def __getitem__(self, idx):
        output = super().__getitem__(idx)
        item = self.data_source[idx]
        osdg_label = item.label
        osdg_label = osdg_label if osdg_label < self.ood_idx else self.ood_idx
        output["osdg_label"] = osdg_label

        if self.is_train:
            # sample img from another domain
            domain = output["domain"]
            domain_selected = random.choice(
                [i for i in range(len(self.data_by_domain)) if i != domain]
            )
            item_selected = random.choice(self.data_by_domain[domain_selected])
            img_s = self.pre_tfms(read_image(item_selected.impath))  # sampled img
            img_o = self.pre_tfms(read_image(item.impath))  # original img
            img_s2o, img_o2s = self.colorful_spectrum_mix(img_o, img_s, alpha=1.0)
            img_s2o = self.post_tfms(img_s2o)
            # img_o2s = self.post_tfms(img_o2s)
            output["img_aug"] = img_s2o

        return output

    def colorful_spectrum_mix(self, img_o, img_s, alpha, ratio=1.0):
        """Input image size: ndarray of [H, W, C]"""
        lam = np.random.uniform(0, alpha)

        assert img_o.shape == img_s.shape
        h, w, c = img_o.shape
        h_crop = int(h * math.sqrt(ratio))
        w_crop = int(w * math.sqrt(ratio))
        h_start = h // 2 - h_crop // 2
        w_start = w // 2 - w_crop // 2

        img_o_fft = np.fft.fft2(img_o, axes=(0, 1))
        img_s_fft = np.fft.fft2(img_s, axes=(0, 1))
        img_o_abs, img_o_pha = np.abs(img_o_fft), np.angle(img_o_fft)
        img_s_abs, img_s_pha = np.abs(img_s_fft), np.angle(img_s_fft)

        img_o_abs = np.fft.fftshift(img_o_abs, axes=(0, 1))
        img_s_abs = np.fft.fftshift(img_s_abs, axes=(0, 1))

        img_o_abs_ = np.copy(img_o_abs)
        img_s_abs_ = np.copy(img_s_abs)
        img_o_abs[h_start : h_start + h_crop, w_start : w_start + w_crop] = (
            lam * img_s_abs_[h_start : h_start + h_crop, w_start : w_start + w_crop]
            + (1 - lam)
            * img_o_abs_[h_start : h_start + h_crop, w_start : w_start + w_crop]
        )
        img_s_abs[h_start : h_start + h_crop, w_start : w_start + w_crop] = (
            lam * img_o_abs_[h_start : h_start + h_crop, w_start : w_start + w_crop]
            + (1 - lam)
            * img_s_abs_[h_start : h_start + h_crop, w_start : w_start + w_crop]
        )

        img_o_abs = np.fft.ifftshift(img_o_abs, axes=(0, 1))
        img_s_abs = np.fft.ifftshift(img_s_abs, axes=(0, 1))

        img_s1 = img_o_abs * (np.e ** (1j * img_o_pha))
        img_o2 = img_s_abs * (np.e ** (1j * img_s_pha))
        img_s1 = np.real(np.fft.ifft2(img_s1, axes=(0, 1)))
        img_o2 = np.real(np.fft.ifft2(img_o2, axes=(0, 1)))
        img_s1 = np.uint8(np.clip(img_s1, 0, 255))
        img_o2 = np.uint8(np.clip(img_o2, 0, 255))

        return img_s1, img_o2


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
            dataset=dataset,
            data_source=train_set,
            ood_idx=ood_idx,
            batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
            n_domain=cfg.DATALOADER.TRAIN_X.N_DOMAIN,
            n_ins=cfg.DATALOADER.TRAIN_X.N_INS,
            tfm=tfm_train,
            is_train=True,
            sampler_type=cfg.DATALOADER.TRAIN_X.SAMPLER,
        )

        # Build val_loader
        val_loader = None
        if dataset.val:
            val_loader = build_osdg_data_loader(
                cfg,
                dataset=dataset,
                data_source=dataset.val,
                ood_idx=ood_idx,
                batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                tfm=tfm_test,
                is_train=False,
                # sampler_type=cfg.DATALOADER.TEST.SAMPLER,
                sampler_type=cfg.DATALOADER.TRAIN_X.SAMPLER,
            )

        # Build test_loader
        test_loader = build_osdg_data_loader(
            cfg,
            dataset=dataset,
            data_source=dataset.test,
            ood_idx=ood_idx,
            batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
            tfm=tfm_test,
            is_train=False,
            sampler_type=cfg.DATALOADER.TEST.SAMPLER,
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


class OSDG_Evaluator(EvaluatorBase):

    def __init__(self, cfg, ood_idx, lab2cname=None):
        self.ood_idx = ood_idx
        self._lab2cname = lab2cname
        self._correct_known = 0
        self._total_known = 0
        self._correct_ood = 0
        self._total_ood = 0
        self._y_max_prob = []
        self._y_true = []
        self._y_pred = []

    def reset(self):
        self.ood_thresh = None
        self._correct_known = 0
        self._total_known = 0
        self._correct_ood = 0
        self._total_ood = 0
        self._y_max_prob = []
        self._y_true = []
        self._y_pred = []

    def process(self, output, osdg_label):
        # output (torch.Tensor): model output [B, C_k + 1]
        # osdg_label (torch.LongTensor): ground truth [B]
        max_probs, y_pred = output.max(1)
        self._y_max_prob.extend(max_probs.data.cpu().numpy().tolist())
        self._y_true.extend(osdg_label.data.cpu().numpy().tolist())
        self._y_pred.extend(y_pred.data.cpu().numpy().tolist())

        known_mask = osdg_label < self.ood_idx
        ood_mask = ~known_mask

        # handle known classes
        known_gt = osdg_label[known_mask]
        known_pred = y_pred[known_mask]
        matches_known = known_pred.eq(known_gt).float()
        self._correct_known += int(matches_known.sum().item())
        self._total_known += known_gt.shape[0]

        # handle OOD classes
        ood_gt = osdg_label[ood_mask]
        ood_pred = y_pred[ood_mask]
        matches_ood = ood_pred.eq(ood_gt).float()
        self._correct_ood += int(matches_ood.sum().item())
        self._total_ood += ood_gt.shape[0]
        # print(ood_gt, ood_pred)
        # IPython.embed()

    def do_evaluate(
        self,
        y_max_prob,
        y_true,
        y_pred,
        correct_known,
        total_known,
        correct_ood,
        total_ood,
    ):
        results = OrderedDict()
        closed_acc = 100.0 * correct_known / total_known
        ood_acc = 100.0 * correct_ood / total_ood
        h_score = (2 * closed_acc * ood_acc) / (closed_acc + ood_acc)

        print(self._correct_ood, self._total_ood)
        results["Acc"] = closed_acc
        results["H-Score"] = h_score

        logger.info(
            "=> result\n"
            f"* Acc_k: {closed_acc:.2f}\n"
            f"* Acc_u: {ood_acc:.2f}\n"
            f"* H-Score: {h_score:.2f}\n"
        )

        return results

    def evaluate(self):
        if comm.get_world_size() > 1:
            comm.synchronize()
            y_max_prob = comm.gather(self._y_max_prob, dst=0)
            y_true = comm.gather(self._y_true, dst=0)
            y_pred = comm.gather(self._y_pred, dst=0)
            correct_known = comm.gather(self._correct_known, dst=0)
            total_known = comm.gather(self._total_known, dst=0)
            correct_ood = comm.gather(self._correct_ood, dst=0)
            total_ood = comm.gather(self._total_ood, dst=0)
            if comm.is_main_process():
                y_true = list(itertools.chain(*y_true))
                y_pred = list(itertools.chain(*y_pred))
                correct_known = sum(correct_known)
                total_known = sum(total_known)
                correct_ood = sum(correct_ood)
                total_ood = sum(total_ood)
                results = [
                    self.do_evaluate(
                        y_max_prob,
                        y_true,
                        y_pred,
                        correct_known,
                        total_known,
                        correct_ood,
                        total_ood,
                    )
                ]
            else:
                results = [None]
            # comm.synchronize()
            dist.broadcast_object_list(results, src=0)
            results = results[0]
            # comm.synchronize()
        else:
            y_max_prob = self._y_max_prob
            y_true = self._y_true
            y_pred = self._y_pred
            correct_known = self._correct_known
            total_known = self._total_known
            correct_ood = self._correct_ood
            total_ood = self._total_ood
            results = self.do_evaluate(
                y_max_prob,
                y_true,
                y_pred,
                correct_known,
                total_known,
                correct_ood,
                total_ood,
            )
        # IPython.embed()
        return results


@TRAINER_REGISTRY.register()
class OSDG_PlBaseTrainer(DGPlBaseTrainer):

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
        self.evaluator = build_evaluator(cfg, lab2cname=self.lab2cname)
        self.osdg_evaluator = OSDG_Evaluator(
            cfg, self.ood_idx, lab2cname=self.lab2cname
        )
        self.best_result = -np.inf

    def build_data_loader(self):
        """Create essential data-related attributes.

        A re-implementation of this method must create the
        same attributes (self.dm is optional).
        """
        dm = OSDG_DataManager(self.cfg)

        self.train_loader = dm.train_loader
        self.val_loader = dm.val_loader  # optional, can be None
        self.test_loader = dm.test_loader

        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname  # dict {label: classname}
        self.ood_idx = dm.ood_idx

        self.dm = dm

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.known_classnames

        logger.info(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAIN.PREC == "fp32" or cfg.TRAIN.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        logger.info("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        logger.info("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        logger.info(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)

        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model(
            "prompt_learner", self.model.prompt_learner, self.optim, self.sched
        )

        self.scaler = GradScaler() if cfg.TRAIN.PREC == "amp" else None

        # For training, wrap with DDP. But don't need this for inference.
        if comm.get_world_size() > 1:
            logger.info("wrap the model with DistributedDataParallel")
            # ref to https://github.com/pytorch/pytorch/issues/22049 to set `find_unused_parameters=True`
            # for part of the parameters is not updated.
            # self.model = DistributedDataParallel(
            #     self.model,
            #     device_ids=[comm.get_local_rank()],
            #     broadcast_buffers=False,
            # )
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[comm.get_local_rank()],
                broadcast_buffers=False,
            )

    def parse_osdg_batch(self, batch, is_train=False):
        input = batch["img"]
        label = batch["label"]
        osdg_label = batch["osdg_label"]
        domain = batch["domain"]
        input = input.to(self.device, non_blocking=True)
        label = label.to(self.device, non_blocking=True)
        osdg_label = osdg_label.to(self.device, non_blocking=True)
        domain = domain.to(self.device, non_blocking=True)
        if is_train:
            input_aug = batch["img_aug"]
            input_aug = input_aug.to(self.device, non_blocking=True)
            return input, input_aug, label, osdg_label, domain
        else:
            return input, label, osdg_label, domain

    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.train_loader)

        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader):
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (self.max_epoch - self.epoch - 1) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                logger.info(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()

    def forward_backward(self, batch):
        images, images_aug, labels, osdg_labels, domains = self.parse_osdg_batch(
            batch, is_train=True
        )

        prec = self.cfg.TRAIN.PREC
        if prec == "amp":
            with autocast():
                loss_dict, logits = self.model(images, osdg_labels, images_aug)
                losses = sum(loss_dict.values())
            self.optim.zero_grad()
            self.scaler.scale(losses).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            loss_dict, logits = self.model(images, osdg_labels, images_aug)
            losses = sum(loss_dict.values())
            self.model_backward_and_update(losses)

        loss_summary = loss_dict
        loss_summary.update(
            {
                "acc": compute_accuracy(logits.detach(), osdg_labels)[0].item(),
            }
        )

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def after_epoch(self):
        self.test(split="test")

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.osdg_evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        logger.info(f"Evaluate on the *{split}* set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            images, labels, osdg_label, domains = self.parse_osdg_batch(batch)
            output = self.model_inference(images)
            self.osdg_evaluator.process(output, osdg_label)

        results = self.osdg_evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]


def set_osdgpl_config(cfg):
    _C = cfg
    # fmt: off
    _C.DATASET.ROOT = "/root/xfb/datasets/DG"
    _C.DATASET.NAME = "PACS"
    _C.DATASET.SOURCE_DOMAINS = ("cartoon", "photo", "sketch")
    _C.DATASET.TARGET_DOMAINS = ("art_painting",)
    # _C.DATASET.SOURCE_DOMAINS = ("art_painting", "photo", "sketch")
    # _C.DATASET.TARGET_DOMAINS = ("cartoon",)
    # _C.DATASET.SOURCE_DOMAINS = ("art_painting", "cartoon","sketch",)
    # _C.DATASET.TARGET_DOMAINS = ("photo",)
    # _C.DATASET.SOURCE_DOMAINS = ("art_painting", "cartoon", "photo")
    # _C.DATASET.TARGET_DOMAINS = ("sketch",)

    # _C.DATASET.NAME = "OfficeHome"
    # _C.DATASET.SOURCE_DOMAINS = ("clipart", "product", "real_world")
    # _C.DATASET.TARGET_DOMAINS = ("art",)

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

    # fmt: on


def set_method_config(cfg):
    _C = cfg
    _C.SEED = 1

    # -----------------------------------------------------------------------------
    # METHOD
    # -----------------------------------------------------------------------------
    _C.METHOD = CN()
    _C.METHOD.T = 1  # temperature
    _C.METHOD.N_CTX = 16  # number of context vectors
    _C.METHOD.N_NEG = 1  #
    _C.METHOD.N_SAMPLE = 1000
    _C.METHOD.N_SELECT = 1
    _C.METHOD.N_SAMPLE_FROM = 10000


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    set_osdgpl_config(cfg)
    set_method_config(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    merge_from_args(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    # 5. clean unused trainer configs
    clean_cfg(cfg, cfg.TRAINER.NAME)

    default_setup(cfg, args)

    cfg.freeze()
    return cfg


def main(args):
    cfg = setup(args)
    trainer = OSDG_PlBaseTrainer(cfg)

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test(split="test")
        return

    if not args.no_train:
        trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    logger.info("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
