import argparse
import math
import os
import os.path as osp
import sys
from collections import OrderedDict
from datetime import datetime

import torch
import torch.nn as nn
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn import functional as F

import libs.data.datasets.caltech101
import libs.data.datasets.domainnet
import libs.data.datasets.dtd
import libs.data.datasets.eurosat
import libs.data.datasets.fgvc_aircraft
import libs.data.datasets.food101
import libs.data.datasets.imagenet
import libs.data.datasets.imagenet_a
import libs.data.datasets.imagenet_r
import libs.data.datasets.imagenet_sketch
import libs.data.datasets.imagenetv2
import libs.data.datasets.mini_domainnet
import libs.data.datasets.office_home
import libs.data.datasets.oxford_flowers
import libs.data.datasets.oxford_pets
import libs.data.datasets.pacs
import libs.data.datasets.stanford_cars
import libs.data.datasets.sun397
import libs.data.datasets.ucf101
import libs.data.datasets.vlcs
from dassl.config import get_cfg_default
from dassl.utils import collect_env_info
from libs.utils import comm
from libs.utils.env import seed_all_rng
from libs.utils.file_io import PathManager
from libs.utils.logger import setup_logger


def default_argument_parser():
    """
    Create a parser with some common arguments used by users.
    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument(
        "--config-file",
        type=str,
        default="",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--root", type=str, help="path to dataset")
    parser.add_argument(
        "--no-log-output", action="store_true", help="not log to output directory"
    )
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="whether to attempt to resume from the checkpoint directory",
    )
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )

    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument(
        "--num-gpus", type=int, default=1, help="number of gpus *per machine*"
    )
    parser.add_argument(
        "--num-machines", type=int, default=1, help="total number of machines"
    )
    parser.add_argument(
        "--machine-rank",
        type=int,
        default=0,
        help="the rank of this machine (unique per machine)",
    )

    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2**15 + 2**14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2**14
    parser.add_argument("--dist-url", default="tcp://127.0.0.1:{}".format(port))
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def merge_from_args(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head


def extend_cfg(cfg):
    """
    Add new config variables.
    """
    from yacs.config import CfgNode as CN

    cfg.CUDNN_BENCHMARK = False

    cfg.TRAIN.PREC = "fp16"  # fp16, fp32, amp

    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COOP.CSC = False  # class-specific context
    cfg.TRAINER.COOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COCOOP.CTX_INIT = ""  # initialization words

    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new


def get_cfg():
    cfg = get_cfg_default()
    extend_cfg(cfg)
    return cfg


def default_setup(cfg, args):
    """
    Perform some basic common setups at the beginning of a job, including:
    1. Set up the detectron2 logger
    2. Log basic information about environment, cmdline arguments, and config
    3. Backup the config to the output directory
    Args:
        cfg (CfgNode): the full config to be used
        args (argparse.NameSpace): the command line arguments to be logged
    """

    if args.no_log_output:
        output_dir = None
    else:
        output_dir = cfg.OUTPUT_DIR
        if comm.is_main_process() and output_dir:
            PathManager.mkdirs(output_dir)

    if cfg.DATASET.ROOT == "":
        cfg.DATASET.ROOT = "/root/work/datasets/"

    rank = comm.get_rank()

    logger = setup_logger(output_dir, distributed_rank=rank)
    logger.info(
        "Rank of current process: {}. World size: {}".format(
            rank, comm.get_world_size()
        )
    )
    logger.info("Environment info:\n" + collect_env_info())

    logger.info("******************************")
    logger.info("** Command line arguments: **")
    logger.info("******************************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        logger.info(f"{key}: {args.__dict__[key]}")

    if hasattr(args, "config_file") and args.config_file != "":
        logger.info(
            f"Contents of args.config_file={args.config_file}:\n"
            f"{PathManager.open(args.config_file, 'r').read()}"
        )

    # make sure each worker has a different, yet deterministic seed if specified
    seed = None
    if args.seed >= 0:
        seed = args.seed
        logger.info(f"using specified seed = {seed}")
    else:
        seed = (
            os.getpid()
            + int(datetime.now().strftime("%S%f"))
            + int.from_bytes(os.urandom(2), "big")
        )
        logger.info(f"Setting random seed = {seed}")
    cfg.SEED = seed
    seed_all_rng(seed)

    logger.info("******************************")
    logger.info("** Running with full config: **")
    logger.info("******************************")
    logger.info(cfg)
    if not args.no_log_output:
        if comm.is_main_process() and output_dir:
            # Note: some of our scripts may expect the existence of
            # config.yaml in output directory
            path = os.path.join(output_dir, "config.yaml")
            with PathManager.open(path, "w") as f:
                f.write(cfg.dump())
            logger.info(f"Full config saved to {os.path.abspath(path)}")

    # cudnn benchmark has large overhead. It shouldn't be used considering the small size of typical validation set.
    if not (hasattr(args, "eval_only") and args.eval_only):
        torch.backends.cudnn.benchmark = cfg.CUDNN_BENCHMARK
