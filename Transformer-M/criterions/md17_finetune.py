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


@register_criterion("md17_force", dataclass=GraphPredictionConfig)
class MD17ForceLoss(FairseqCriterion):
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

        if self.noise_scale == 0.0:
            model_output = model(**sample["net_input"])
            logits, force_pred, node_output, _ = model_output
            logits = logits[:,0,:]
            # energy_targets = sample["net_input"]["batched_data"]['y'].unsqueeze(-1)
            force_targets = sample["net_input"]["batched_data"]['dy'].reshape(sample_size, -1, 3)

            if self.loss_type == 'L1':
                loss_func = nn.L1Loss
            elif self.loss_type == 'L2':
                loss_func = nn.MSELoss
            else:
                raise NotImplementedError

            force_loss = loss_func(reduction="mean")(force_targets, force_pred)
            loss = force_loss

            logging_output = {
                "loss": loss.data,
                "force_loss": force_loss.data,
                "sample_size": sample_size,
                "nsentences": sample_size,
                "ntokens": natoms,
            }
            return loss, sample_size, logging_output
        
        else:
            ori_pos = sample['net_input']['batched_data']['pos']
            noise = torch.randn(ori_pos.shape).to(ori_pos) * self.noise_scale
            sample['net_input']['batched_data']['pos'] = ori_pos + noise

            model_output = model(**sample["net_input"])
            logits, force_pred, node_output, _ = model_output
            logits = logits[:,0,:]
            # energy_targets = sample["net_input"]["batched_data"]['y'].unsqueeze(-1)
            force_targets = sample["net_input"]["batched_data"]['dy'].reshape(sample_size, -1, 3)

            if self.loss_type == 'L1':
                loss_func = nn.L1Loss
            elif self.loss_type == 'L2':
                loss_func = nn.MSELoss
            else:
                raise NotImplementedError
            
            force_loss = loss_func(reduction="mean")(force_targets, force_pred)
            pos_loss = (1.0 - nn.CosineSimilarity(dim=-1)(node_output.to(torch.float32), noise.to(torch.float32))).mean()
            loss = force_loss + pos_loss

            logging_output = {
                "loss": loss.data,
                "force_loss": force_loss.data,
                "pos_loss": pos_loss.data,
                "sample_size": sample_size,
                "nsentences": sample_size,
                "ntokens": natoms,
            }
            return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        force_loss_sum = sum(log.get("force_loss", 0) for log in logging_outputs)
        pos_loss_sum = sum(log.get("pos_loss", 0) for log in logging_outputs)
        energy_loss_sum = sum(log.get("energy_loss", 0) for log in logging_outputs)
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)

        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        # metrics.log_scalar(
            # "loss", loss_sum / sample_size, sample_size, round=6
        # )

        metrics.log_scalar("loss", loss_sum, sample_size, round=6)
        metrics.log_scalar("force_loss", force_loss_sum, sample_size, round=6)
        metrics.log_scalar("pos_loss", pos_loss_sum, sample_size, round=6)
        metrics.log_scalar("energy_loss", energy_loss_sum, sample_size, round=6)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True