from dataclasses import dataclass
import math
from omegaconf import II

import torch
import torch.nn as nn
from fairseq import metrics, modules, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
import os

from sklearn.metrics import roc_auc_score


@dataclass
class GraphPredictionConfig(FairseqDataclass):
    tpu: bool = II("common.tpu")


@register_criterion("molnet_finetune", dataclass=GraphPredictionConfig)
class MolNetFinetuneLoss(FairseqCriterion):
    """
    Implementation for the loss used in masked graph model (MGM) training.
    """

    def __init__(self, cfg: GraphPredictionConfig, task):
        super().__init__(task)
        self.tpu = cfg.tpu
        self.noise_scale = task.cfg.noise_scale
        self.loss_type = task.cfg.loss_type

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        sample_size = sample["nsamples"]

        with torch.no_grad():
            natoms = sample["net_input"]["batched_data"]['x'].shape[1]

        model_output = model(**sample["net_input"])
        logits, node_output = model_output[0], model_output[1]
        logits = logits[:,0,:]
        targets = sample["net_input"]["batched_data"]['y'].unsqueeze(-1)
        # targets = model.get_targets(sample, [logits])

        mean = sample['net_input']['batched_data']['mean']
        std = sample['net_input']['batched_data']['std']
        
        new_labels = targets
        new_logits = logits * std + mean # 1, 0 for classification tasks

        if self.loss_type == 'L1':
            loss_func = nn.L1Loss
        elif self.loss_type == 'L2':
            loss_func = nn.MSELoss
        elif self.loss_type == 'SmoothL1':
            loss_func = nn.SmoothL1Loss
        elif self.loss_type == 'BCELogits':
            nan_mask = torch.isnan(targets)
            loss_func = nn.BCEWithLogitsLoss
            new_logits, new_labels = new_logits.reshape(-1,1)[~nan_mask], new_labels[~nan_mask]
        # import numpy as np
        # import time
        # import pdb; pdb.set_trace()
        # np.save(f"bbbp_test_batch{time.time()}", model_output[2]['inner_states'][-1].permute(1,0,2)[:,0,:].detach().cpu().numpy())
        # np.save(f"bbbp_test_label_batch{time.time()}", targets.detach().cpu().numpy())

        loss = loss_func(reduction="mean")(new_logits, new_labels)
        if self.loss_type == 'BCELogits':
            try:
                eval_loss = roc_auc_score(new_labels.detach().cpu(), new_logits.detach().cpu())
            except ValueError:
                import numpy as np
                assert len(np.unique(new_labels.detach().cpu())) == 1, "new_labels contains only one class"
                eval_loss = 1.0
        else:
            eval_loss = nn.L1Loss(reduction='mean')(new_logits, new_labels)
            eval_loss = eval_loss.data

        logging_output = {
            "loss": loss.data,
            "eval_loss": eval_loss,
            "sample_size": sample_size,
            "nsentences": sample_size,
            "ntokens": natoms,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        eval_loss_sum = sum(log.get("eval_loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        # metrics.log_scalar(
            # "loss", loss_sum / sample_size, sample_size, round=6
        # )
        # metrics.log_scalar(
            # "eval_loss", eval_loss_sum / sample_size, sample_size, round=6
        # )

        metrics.log_scalar(
            "loss", loss_sum, sample_size, round=6
        )
        metrics.log_scalar(
            "eval_loss", eval_loss_sum, sample_size, round=6
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True