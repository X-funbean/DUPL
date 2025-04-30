import glob
import logging
import os
import os.path as osp

from dassl.data.datasets import DATASET_REGISTRY, DatasetBase, Datum
from dassl.utils import listdir_nohidden, mkdir_if_missing

logger = logging.getLogger(
    f'fastdg.{os.path.relpath(__file__).replace(os.path.sep, ".")}'
)


@DATASET_REGISTRY.register()
class VLCS(DatasetBase):
    """VLCS.

    Statistics:
        - 4 domains: CALTECH, LABELME, PASCAL, SUN
        - 5 categories: bird, car, chair, dog, and person.

    Reference:
        - Torralba and Efros. Unbiased look at dataset bias. CVPR 2011.
    """

    dataset_dir = "VLCS"
    domains = ["caltech", "labelme", "pascal", "sun"]
    class_names = ["bird", "car", "chair", "dog", "person"]
    data_url = "https://drive.google.com/uc?id=1r0WL5DDqKfSPp9E3tRENwHaXNs1olLZd"

    def __init__(self, cfg, seed=None):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)

        if not osp.exists(self.dataset_dir):
            dst = osp.join(root, "vlcs.zip")
            self.download_data(self.data_url, dst, from_gdrive=True)

        self.src_domains = cfg.DATASET.SOURCE_DOMAINS
        self.tgt_domains = cfg.DATASET.TARGET_DOMAINS
        self.check_input_domains(self.src_domains, self.tgt_domains)

        if cfg.DOMAINBED.USE_FIXED_SPLIT:
            train = self._read_data(cfg.DATASET.SOURCE_DOMAINS, "train")
            val = self._read_data(cfg.DATASET.SOURCE_DOMAINS, "crossval")
            test = self._read_data(cfg.DATASET.TARGET_DOMAINS, "full")
            all_items = train + val + test
        else:
            src_items = self._read_data(cfg.DATASET.SOURCE_DOMAINS, "full")
            tgt_items = self._read_data(cfg.DATASET.TARGET_DOMAINS, "full")
            all_items = src_items + tgt_items

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

    def _read_data(self, input_domains, split):
        items = []

        for domain, dname in enumerate(sorted(input_domains)):
            path = osp.join(self.dataset_dir, dname.upper(), split)
            folders = listdir_nohidden(path)
            folders.sort()

            for label, folder in enumerate(folders):
                impaths = glob.glob(osp.join(path, folder, "*.jpg"))

                for impath in impaths:
                    item = Datum(
                        impath=impath,
                        label=label,
                        domain=domain,
                        classname=self.class_names[label],
                        domain_name=dname,
                    )
                    items.append(item)

        return items
