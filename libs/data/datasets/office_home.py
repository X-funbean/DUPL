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
class OfficeHome(DatasetBase):
    """Office-Home.

    Statistics:
        - Around 15,500 images.
        - 65 classes related to office and home objects.
        - 4 domains: Art, Clipart, Product, Real World.
        - URL: http://hemanthdv.org/OfficeHome-Dataset/.

    Reference:
        - Venkateswara et al. Deep Hashing Network for Unsupervised
        Domain Adaptation. CVPR 2017.
    """

    dataset_dir = "office_home_dg"
    domains = ["art", "clipart", "product", "real_world"]
    data_url = "https://drive.google.com/uc?id=1gkbf_KaxoBws-GWT3XIPZ7BnkqbAxIFa"

    def __init__(self, cfg, seed=None):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)

        if not osp.exists(self.dataset_dir):
            dst = osp.join(root, "office_home_dg.zip")
            self.download_data(self.data_url, dst, from_gdrive=True)

        self.src_domains = cfg.DATASET.SOURCE_DOMAINS
        self.tgt_domains = cfg.DATASET.TARGET_DOMAINS
        self.check_input_domains(self.src_domains, self.tgt_domains)

        if cfg.DOMAINBED.USE_FIXED_SPLIT:
            train = self._read_data(cfg.DATASET.SOURCE_DOMAINS, "train")
            val = self._read_data(cfg.DATASET.SOURCE_DOMAINS, "val")
            test = self._read_data(cfg.DATASET.TARGET_DOMAINS, "all")
            all_items = train + val + test
        else:
            src_items = self._read_data(cfg.DATASET.SOURCE_DOMAINS, "all")
            tgt_items = self._read_data(cfg.DATASET.TARGET_DOMAINS, "all")
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

        def _load_data_from_directory(directory):
            folders = listdir_nohidden(directory)
            folders.sort()
            items_ = []

            for label, folder in enumerate(folders):
                impaths = glob.glob(osp.join(directory, folder, "*.jpg"))

                for impath in impaths:
                    items_.append((impath, label))

            return items_

        items = []

        for domain, dname in enumerate(sorted(input_domains)):
            if split == "all":
                train_dir = osp.join(self.dataset_dir, dname, "train")
                impath_label_list = _load_data_from_directory(train_dir)
                val_dir = osp.join(self.dataset_dir, dname, "val")
                impath_label_list += _load_data_from_directory(val_dir)
            else:
                split_dir = osp.join(self.dataset_dir, dname, split)
                impath_label_list = _load_data_from_directory(split_dir)

            for impath, label in impath_label_list:
                class_name = impath.split("/")[-2].lower()
                item = Datum(
                    impath=impath,
                    label=label,
                    domain=domain,
                    classname=class_name,
                    domain_name=dname,
                )
                items.append(item)

        return items
