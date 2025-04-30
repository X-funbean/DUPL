import logging
import os
import os.path as osp

from dassl.data.datasets import DATASET_REGISTRY, DatasetBase, Datum
from dassl.utils import listdir_nohidden, mkdir_if_missing

logger = logging.getLogger(
    f'fastdg.{os.path.relpath(__file__).replace(os.path.sep, ".")}'
)


@DATASET_REGISTRY.register()
class miniDomainNet(DatasetBase):
    """The DomainNet multi-domain data loader

    Statistics:
        - 4 distinct domains: Clipart, Painting, Real, Sketch.
        - Around 0.6M images.
        - 345 categories.
        - URL: http://ai.bu.edu/M3SDA/, cleaned version.

    Reference:
        - Peng et al. Moment Matching for Multi-Source Domain Adaptation. ICCV 2019.
        - Zhou et al. Domain Adaptive Ensemble Learning. TIP 2021.
    """

    dataset_dir = "DomainNet"
    # domains = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
    domains = ["clipart", "painting", "real", "sketch"]

    def __init__(self, cfg, seed=None):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)

        self.src_domains = cfg.DATASET.SOURCE_DOMAINS
        self.tgt_domains = cfg.DATASET.TARGET_DOMAINS
        self.check_input_domains(self.src_domains, self.tgt_domains)

        src_items = self._read_data(
            cfg.DATASET.SOURCE_DOMAINS, split="train"
        ) + self._read_data(cfg.DATASET.SOURCE_DOMAINS, split="test")
        tgt_items = self._read_data(
            cfg.DATASET.TARGET_DOMAINS, split="train"
        ) + self._read_data(cfg.DATASET.TARGET_DOMAINS, split="test")
        all_items = src_items + tgt_items

        if cfg.DOMAINBED.USE_FIXED_SPLIT:
            train = src_items
            val = []
            test = tgt_items
            all_items = train + val + test
        else:
            holdout_fraction = cfg.DOMAINBED.HOLDOUT_FRACTION
            train, val, test = self.random_split_in_out_set(
                src_items, tgt_items, holdout_fraction, seed
            )

        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            logger.info(
                f"generating {num_shots}-shots train sets, {min(num_shots, 4)}-shots val sets"
            )
            train = self.generate_fewshot_dataset(train, num_shots=num_shots)
            val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))

        super().__init__(train_x=train, val=val, test=test, all_items=all_items)

    def _read_data(self, input_domains, split="train"):
        items = []

        for domain, dname in enumerate(sorted(input_domains)):
            filename = f"{dname}_{split}.txt"
            split_file = osp.join(self.dataset_dir, 'splits_mini', filename)

            with open(split_file, "r") as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    impath, label = line.split(" ")
                    classname = impath.split("/")[1]
                    impath = osp.join(self.dataset_dir, impath)
                    label = int(label)
                    item = Datum(
                        impath=impath,
                        label=label,
                        domain=domain,
                        classname=classname,
                        domain_name=dname,
                    )
                    items.append(item)

        return items

    def get_num_classes(self, data_source):
        return 126
