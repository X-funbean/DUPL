# import itertools
# import logging
# import os
# import os.path as osp
# from collections import OrderedDict, defaultdict

# import numpy as np
# import torch
# import torch.distributed as dist
# from sklearn.metrics import confusion_matrix, f1_score

# from dassl.evaluation import EVALUATOR_REGISTRY, EvaluatorBase
# from libs.utils import comm

# logger = logging.getLogger(
#     f'fastdg.{os.path.relpath(__file__).replace(os.path.sep, ".")}'
# )


# @EVALUATOR_REGISTRY.register()
# class DGClassification(EvaluatorBase):

#     def __init__(self, cfg, mode, **kwargs):
#         super().__init__(cfg)
#         self._correct = 0
#         self._total = 0
#         self._per_class_res = None
#         self._y_true = []
#         self._y_pred = []
#         self._y_gt_dom = []

#         if cfg.TEST.PER_TARGET_RESULT:
#             self._dom_names = sorted(dom_names)
#             self._per_target_res = defaultdict(list)

#     def reset(self):
#         self._correct = 0
#         self._total = 0
#         self._y_true = []
#         self._y_pred = []
#         self._y_gt_dom = []
#         self._per_target_res = defaultdict(list)

#     def process(self, mo, gt, domains):
#         # mo (torch.Tensor): model output [B, C]
#         # gt (torch.LongTensor): ground truth [B]
#         pred = mo.max(1)[1]
#         matches = pred.eq(gt).float()
#         self._correct += int(matches.sum().item())
#         self._total += gt.shape[0]

#         self._y_true.extend(gt.data.cpu().numpy().tolist())
#         self._y_pred.extend(pred.data.cpu().numpy().tolist())
#         self._y_gt_dom.extend(domains.data.cpu().numpy().tolist())

#     def do_evaluate(self, y_true, y_pred, correct, total):
#         results = OrderedDict()
#         acc = 100.0 * correct / total
#         err = 100.0 - acc
#         macro_f1 = 100.0 * f1_score(
#             y_true, y_pred, average="macro", labels=np.unique(self._y_true)
#         )

#         # The first value will be returned by trainer.test()
#         results["accuracy"] = acc
#         results["error_rate"] = err
#         results["macro_f1"] = macro_f1

#         if cfg.TEST.PER_TARGET_RESULT:

#         logger.info(
#             "=> result\n"
#             f"* total: {self._total:,}\n"
#             f"* correct: {self._correct:,}\n"
#             f"* accuracy: {acc:.1f}%\n"
#             f"* error: {err:.1f}%\n"
#             f"* macro_f1: {macro_f1:.1f}%"
#         )

#         if self._per_class_res is not None:
#             labels = list(self._per_class_res.keys())
#             labels.sort()

#             logger.info("=> per-class result")
#             accs = []

#             for label in labels:
#                 classname = self._lab2cname[label]
#                 res = self._per_class_res[label]
#                 correct = sum(res)
#                 total = len(res)
#                 acc = 100.0 * correct / total
#                 accs.append(acc)
#                 logger.info(
#                     f"* class: {label} ({classname})\t"
#                     f"total: {total:,}\t"
#                     f"correct: {correct:,}\t"
#                     f"acc: {acc:.1f}%"
#                 )
#             mean_acc = np.mean(accs)
#             logger.info(f"* average: {mean_acc:.1f}%")

#             results["perclass_accuracy"] = mean_acc

#         if self.cfg.TEST.COMPUTE_CMAT:
#             cmat = confusion_matrix(self._y_true, self._y_pred, normalize="true")
#             save_path = osp.join(self.cfg.OUTPUT_DIR, "cmat.pt")
#             torch.save(cmat, save_path)
#             logger.info(f"Confusion matrix is saved to {save_path}")

#         return results

#     def evaluate(self):
#         if comm.get_world_size() > 1:
#             comm.synchronize()
#             y_true = comm.gather(self._y_true, dst=0)
#             y_pred = comm.gather(self._y_pred, dst=0)
#             correct = comm.gather(self._correct, dst=0)
#             total = comm.gather(self._total, dst=0)
#             # TODO: per_class_res

#             if comm.is_main_process():
#                 y_true = list(itertools.chain(*y_true))
#                 y_pred = list(itertools.chain(*y_pred))
#                 correct = sum(correct)
#                 total = sum(total)
#                 results = [self.do_evaluate(y_true, y_pred, correct, total)]
#             else:
#                 results = [None]
#             # comm.synchronize()
#             dist.broadcast_object_list(results, src=0)
#             results = results[0]
#             # comm.synchronize()
#         else:
#             y_true = self._y_true
#             y_pred = self._y_pred
#             correct = self._correct
#             total = self._total
#             results = self.do_evaluate(y_true, y_pred, correct, total)

#         return results


