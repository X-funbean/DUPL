import logging
import os
import os.path as osp
import pickle

from dassl.data.datasets import DATASET_REGISTRY, DatasetBase, Datum
from dassl.utils import mkdir_if_missing

logger = logging.getLogger(
    f'fastdg.{os.path.relpath(__file__).replace(os.path.sep, ".")}'
)


@DATASET_REGISTRY.register()
class PACS(DatasetBase):
    """PACS.

    Statistics:
        - 4 domains:
            Photo (1,670), Art (2,048), Cartoon (2,344), Sketch (3,929).
        - 7 categories:
            dog, elephant, giraffe, guitar, horse, house and person.

    Reference:
        - Li et al. Deeper, broader and artier domain generalization. ICCV 2017.
    """

    dataset_dir = "PACS"
    domains = ["art_painting", "cartoon", "photo", "sketch"]
    data_url = "https://drive.google.com/uc?id=1m4X4fROCCXMO0lRLrr6Zz9Vb3974NWhE"
    # the following images contain errors and should be ignored
    _error_paths = ["sketch/dog/n02103406_4068-1.png"]

    def __init__(self, cfg, seed=None):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.image_dir = osp.join(self.dataset_dir, "kfold")
        self.split_dir = osp.join(self.dataset_dir, "splits")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if not osp.exists(self.dataset_dir):
            dst = osp.join(root, "pacs.zip")
            self.download_data(self.data_url, dst, from_gdrive=True)

        self.src_domains = cfg.DATASET.SOURCE_DOMAINS
        self.tgt_domains = cfg.DATASET.TARGET_DOMAINS
        self.check_input_domains(self.src_domains, self.tgt_domains)

        if cfg.DOMAINBED.USE_FIXED_SPLIT:
            train = self._read_data(self.src_domains, "train")
            val = self._read_data(self.src_domains, "crossval")
            test = self._read_data(self.tgt_domains, "all")
            all_items = train + val + test
        else:
            src_items = self._read_data(self.src_domains, "all")
            tgt_items = self._read_data(self.tgt_domains, "all")
            all_items = src_items + tgt_items
            holdout_fraction = cfg.DOMAINBED.HOLDOUT_FRACTION
            train, val, test = self.random_split_in_out_set(
                src_items, tgt_items, holdout_fraction, seed
            )

        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            preprocessed = os.path.join(
                self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl"
            )

            if os.path.exists(preprocessed):
                logger.info(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                logger.info(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

            # logger.info(
            #     f"generating {num_shots}-shots train sets, {min(num_shots, 4)}-shots val sets"
            # )
            # train = self.generate_fewshot_dataset(train, num_shots=num_shots)
            # val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))

        super().__init__(train_x=train, val=val, test=test, all_items=all_items)

    def _read_data(self, input_domains, split):
        items = []

        for domain, dname in enumerate(sorted(input_domains)):
            if split == "all":
                file_train = osp.join(self.split_dir, f"{dname}_train_kfold.txt")
                impath_label_list = self._read_split_pacs(file_train)
                file_val = osp.join(self.split_dir, f"{dname}_crossval_kfold.txt")
                impath_label_list += self._read_split_pacs(file_val)
            else:
                file = osp.join(self.split_dir, f"{dname}_{split}_kfold.txt")
                impath_label_list = self._read_split_pacs(file)

            for impath, label in impath_label_list:
                classname = impath.split("/")[-2]
                domain_name = impath.split("/")[-3]
                item = Datum(
                    impath=impath,
                    label=label,
                    domain=domain,
                    classname=classname,
                    domain_name=domain_name,
                )
                items.append(item)

        return items

    def _read_split_pacs(self, split_file):
        items = []

        with open(split_file, "r") as f:
            lines = f.readlines()

            for line in lines:
                line = line.strip()
                impath, label = line.split(" ")
                if impath in self._error_paths:
                    continue
                impath = osp.join(self.image_dir, impath)
                label = int(label) - 1
                items.append((impath, label))

        return items
