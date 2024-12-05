from dataclasses import dataclass
import math
from omegaconf import II

import torch
import torch.nn as nn
from fairseq import metrics, modules, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass


@dataclass
class LEGOLossConfig(FairseqDataclass):
    tpu: bool = II("common.tpu")


@register_criterion("lego_loss", dataclass=LEGOLossConfig)
class LEGOPretrainLoss(FairseqCriterion):
    """
    Implementation for the loss used in masked graph model (MGM) training.
    """

    def __init__(self, cfg: LEGOLossConfig, task):
        super().__init__(task)
        self.tpu = cfg.tpu

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
            central_mask = sample["net_input"]["batched_data"]["central_mask"]
            # all_mask = sample["net_input"]["batched_data"]["all_mask"]
            masked_edges = sample["net_input"]["batched_data"]["masked_edges"]

        label_pos = sample['net_input']['batched_data']['pos']
        noise_mask = (label_pos == 0.0).all(dim=-1, keepdim=True)

        model_output = model(**sample["net_input"])
        logits, node_output = model_output[0], model_output[1]
        logits = logits[:,0,:]
        targets = model.get_targets(sample, [logits])
        
        target_loss = nn.L1Loss(reduction='sum')(logits, targets)

        node_output = node_output.masked_fill_(noise_mask, 0)

        node_output_flatten = node_output.reshape(-1, 3)
        label_pos_flatten = label_pos.reshape(-1, 3)
        src, dst = masked_edges[0,:], masked_edges[1,:]
        noisy_edges = torch.gather(node_output_flatten, 0, dst.unsqueeze(-1).repeat(1,3).to(torch.int64)) - \
                      torch.gather(node_output_flatten, 0, src.unsqueeze(-1).repeat(1,3).to(torch.int64))  # [len, 3]
        label_edges = torch.gather(label_pos_flatten, 0, dst.view(-1,1).expand(-1,3).to(torch.int64)) - \
                      torch.gather(label_pos_flatten, 0, src.view(-1,1).expand(-1,3).to(torch.int64))

        if model.args.lego_loss_fn == "mse":
            central_loss = ((label_pos[central_mask] - node_output[central_mask])**2).sum(dim=-1).mean().to(label_pos.dtype)
            edge_loss = ((label_edges - noisy_edges)**2).sum(dim=-1).to(label_pos.dtype).mean()
            central_angle_loss, central_length_loss = torch.tensor(0), torch.tensor(0)
            edge_angle_loss, edge_length_loss = torch.tensor(0), torch.tensor(0)

        elif model.args.lego_loss_fn == "cos":
            central_angle_loss = (1.0 - nn.CosineSimilarity(dim=-1)(label_pos[central_mask], node_output[central_mask])).mean().to(label_pos.dtype)
            central_length_loss = ((label_pos[central_mask].norm(dim=-1) - node_output[central_mask].norm(dim=-1))**2).mean()
            edge_angle_loss = (1.0 - nn.CosineSimilarity(dim=-1)(label_edges, noisy_edges)).mean().to(label_pos.dtype)
            edge_length_loss = ((label_edges.norm(dim=-1) - noisy_edges.norm(dim=-1)) ** 2).mean()
            central_loss = (central_angle_loss + central_length_loss).to(label_pos.dtype)
            edge_loss = (edge_angle_loss + edge_length_loss).to(label_pos.dtype)

        else:
            raise NotImplementedError

        logging_output = {
            "target_loss": target_loss.data,
            "central_loss": central_loss.data,
            "edge_loss": edge_loss.data,
            "central_angle_loss": central_angle_loss.data,
            "central_length_loss": central_length_loss.data,
            "edge_angle_loss": edge_angle_loss.data,
            "edge_length_loss": edge_length_loss,
            "loss": central_loss.data + edge_loss.data,
            "sample_size": sample_size,
            "nsentences": sample_size,
            "ntokens": natoms,
        }

        return central_loss + edge_loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        target_loss_sum = sum(log.get("target_loss", 0) for log in logging_outputs)
        central_loss_sum = sum(log.get("central_loss", 0) for log in logging_outputs)
        central_angle_loss_sum = sum(log.get("central_angle_loss", 0) for log in logging_outputs)
        central_length_loss_sum = sum(log.get("central_length_loss", 0) for log in logging_outputs)
        edge_loss_sum = sum(log.get("edge_loss", 0) for log in logging_outputs)
        edge_angle_loss_sum = sum(log.get("edge_angle_loss", 0) for log in logging_outputs)
        edge_length_loss_sum = sum(log.get("edge_length_loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        
        metrics.log_scalar(
            "loss", loss_sum / sample_size, sample_size, round=6
        )
        metrics.log_scalar(
            "target_loss", target_loss_sum / sample_size, sample_size, round=6
        )
        metrics.log_scalar(
            "central_loss", central_loss_sum / sample_size, sample_size, round=6
        )
        metrics.log_scalar(
            "central_angle_loss", central_angle_loss_sum / sample_size, sample_size, round=6
        )
        metrics.log_scalar(
            "central_length_loss", central_length_loss_sum / sample_size, sample_size, round=6
        )
        metrics.log_scalar(
            "edge_loss", edge_loss_sum / sample_size, sample_size, round=6
        )
        metrics.log_scalar(
            "edge_angle_loss", edge_angle_loss_sum / sample_size, sample_size, round=6
        )
        metrics.log_scalar(
            "edge_length_loss", edge_length_loss_sum / sample_size, sample_size, round=6
        )


    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True