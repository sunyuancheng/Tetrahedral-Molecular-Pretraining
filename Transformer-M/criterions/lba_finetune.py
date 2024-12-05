from dataclasses import dataclass
import math
from omegaconf import II

import torch
import torch.nn as nn
from fairseq import metrics, modules, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
import os

from scipy.stats import pearsonr, spearmanr


@dataclass
class GraphPredictionConfig(FairseqDataclass):
    tpu: bool = II("common.tpu")


@register_criterion("lba_finetune", dataclass=GraphPredictionConfig)
class LBAFinetuneLoss(FairseqCriterion):
    """
    Implementation for the loss used in masked graph model (MGM) training.
    """

    def __init__(self, cfg: GraphPredictionConfig, task):
        super().__init__(task)
        self.tpu = cfg.tpu
        self.loss_type = task.cfg.loss_type

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        sample_size = sample["nsamples"]

        if sample_size == 0:
            exit()
        with torch.no_grad():
            natoms = sample["net_input"]["batched_data"]['x'].shape[1]
    
        model_output = model(**sample["net_input"])

        logits = model_output[0]
        logits = logits[:, 0, :]
    
        if self.loss_type == "L1":
            loss_func = nn.L1Loss
        elif self.loss_type == "L2":
            loss_func = nn.MSELoss
        elif self.loss_type == "SmoothL1":
            loss_func = nn.SmoothL1Loss
        else:
            raise NotImplementedError
        
        targets = sample["net_input"]["batched_data"]["y"].unsqueeze(-1)

        mean = sample['net_input']['batched_data']['mean']
        std = sample['net_input']['batched_data']['std']

        new_labels = targets
        new_logits = logits * std + mean
        
        loss = loss_func(reduction="mean")(new_labels, new_logits)
        eval_rmse = torch.sqrt(loss)
        eval_spearman = spearmanr(new_labels.squeeze(-1).cpu().numpy(), new_logits.squeeze(-1).detach().cpu().numpy())[0]
        eval_pearson = pearsonr(new_labels.squeeze(-1).cpu().numpy(), new_logits.squeeze(-1).detach().cpu().numpy())[0]

        logging_output = {
            "loss": loss.data,
            "eval_rmse": eval_rmse.data,
            "eval_spearman": eval_spearman,
            "eval_pearson": eval_pearson,
            "sample_size": sample_size,
            "nsentences": sample_size,
            "ntokens": natoms,
        }
        return loss, sample_size, logging_output


    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        eval_rmse_sum = sum(log.get("eval_rmse", 0) for log in logging_outputs)
        eval_spearman_sum = sum(log.get("eval_spearman", 0) for log in logging_outputs)
        eval_pearson_sum = sum(log.get("eval_pearson", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum, sample_size, round=6
        )
        metrics.log_scalar(
            "eval_rmse", eval_rmse_sum, sample_size, round=6
        )
        metrics.log_scalar(
            "eval_spearman", eval_spearman_sum, sample_size, round=6
        )
        metrics.log_scalar(
            "eval_pearson", eval_pearson_sum, sample_size, round=6
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True