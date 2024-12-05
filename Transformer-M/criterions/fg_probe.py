from dataclasses import dataclass
import math
from omegaconf import II

import torch
import torch.nn as nn
from fairseq import metrics, modules, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
import os


@dataclass
class ProbeTaskConfig(FairseqDataclass):
    tpu: bool = II("common.tpu")


@register_criterion("fg_probe_classification", dataclass=ProbeTaskConfig)
class FGProbeClassification(FairseqCriterion):
    """
    Implementation for the loss used in masked graph model (MGM) training.
    """

    def __init__(self, cfg: ProbeTaskConfig, task):
        super().__init__(task)
        self.tpu = cfg.tpu
        self.loss_type = task.cfg.loss_type
        self.std_type = task.cfg.std_type
        self.readout_type = task.cfg.readout_type

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

        logits = model(**sample["net_input"])[0]
        logits_padding = sample['net_input']['batched_data']['x'][:, :, 0].eq(0).unsqueeze(-1)

        if self.readout_type == 'cls':
            logits = logits[:, 0, :]
        elif self.readout_type == "atomwise":
            logits = logits[:, 1:, :].masked_fill_(logits_padding, 0.0)
        else:
            raise NotImplementedError

        targets = model.get_targets(sample, [logits])

        # mean = sample['net_input']['batched_data']['mean']
        # std = sample['net_input']['batched_data']['std']

        if sample['net_input']['batched_data']['type'] == 'train':
            loss_func = nn.L1Loss if self.loss_type == 'L1' else nn.MSELoss
            loss = loss_func(reduction="sum")(new_logits, new_labels)
        else:
            loss = loss_l1

        logging_output = {
            "loss": loss.data,
            "loss_l1": loss_l1.data,
            "sample_size": sample_size,
            "nsentences": sample_size,
            "ntokens": natoms,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        loss_l1_sum = sum(log.get("loss_l1", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size, sample_size, round=9
        )
        metrics.log_scalar(
            "loss_l1_sum", loss_l1_sum / sample_size, sample_size, round=9
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True