import torch
import numpy as np
from fairseq import checkpoint_utils, utils, options, tasks
from fairseq.logging import progress_bar
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
import ogb
import os
from pathlib import Path
from sklearn.metrics import roc_auc_score

import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from utils import load_pretrained_model

import logging


def eval_probe(args, use_pretrained, checkpoint_path=None, logger=None):
    cfg = convert_namespace_to_omegaconf(args)
    np.random.seed(cfg.common.seed)
    utils.set_torch_seed(cfg.common.seed)

    # initialize task
    task = tasks.setup_task(cfg.task)
    model = task.build_model(cfg.model)
    # print(model)

    # load checkpoint
    if use_pretrained:
        model_state = load_pretrained_model(cfg.task.pretrained_model_name)
    else:
        model_state = torch.load(checkpoint_path)["model"]

    model.load_state_dict(
        model_state, strict=True, model_cfg=cfg.model
    )
    del model_state

    model.to(torch.cuda.current_device())
    print(model)
    # load dataset
    split = args.split
    task.load_dataset(split)
    batch_iterator = task.get_batch_iterator(
        dataset=task.dataset(split),
        max_tokens=cfg.dataset.max_tokens_valid,
        max_sentences=cfg.dataset.batch_size_valid,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            model.max_positions(),
        ),
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
        seed=cfg.common.seed,
        num_workers=cfg.dataset.num_workers,
        epoch=0,
        data_buffer_size=cfg.dataset.data_buffer_size,
        disable_iterator_cache=False,
    )
    itr = batch_iterator.next_epoch_itr(
        shuffle=False, set_dataset_epoch=False
    )
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple")
    )

    # infer
    y_pred = []
    y_true = []
    with torch.no_grad():
        model.eval()
        for i, sample in enumerate(progress):
            sample = utils.move_to_cuda(sample)
            y = model(**sample["net_input"])
            y = y.reshape(-1)
            y_pred.extend(y.detach().cpu())
            y_true.extend(sample["target"].detach().cpu().reshape(-1)[:y.shape[0]])
            torch.cuda.empty_cache()
    
    
    y_pred = torch.Tensor(y_pred)
    y_true = torch.Tensor(y_true)

    # rescale
    if hasattr(task.dm.dataset, "task_idx"): #qm9
        train_mean, train_std = sample['net_input']['batched_data']['mean'], sample['net_input']['batched_data']['std']
        y_pred_rescaled = y_pred * train_std + train_mean
        if args.metric == "mae":
            mae = np.mean(np.abs(y_true.numpy() - y_pred_rescaled.numpy()))
            print(f"mae: {mae}")
            return mae
        else:
            raise ValueError(f"Unsupported metric {args.metric}")
        

    # evaluate pretrained models
    if args.metric == "auc":
        nan_mask = torch.isnan(y_true)
        auc = roc_auc_score(y_true[~nan_mask], y_pred[~nan_mask])
        print(f"auc: {auc}")
        return auc
    elif args.metric == "mae":
        mae = np.mean(np.abs(y_true.numpy() - y_pred.numpy()))
        print(f"mae: {mae}")
        return mae
    else:
        raise ValueError(f"Unsupported metric {args.metric}")


def main():
    parser = options.get_training_parser()
    parser.add_argument(
        "--split",
        type=str,
    )
    parser.add_argument(
        "--metric",
        type=str,
    )
    args = options.parse_args_and_arch(parser, modify_parser=None)
    logger = logging.getLogger(__name__)
    if args.pretrained_model_name != "none":
        eval_probe(args, True, logger=logger)
    elif hasattr(args, "save_dir"):
        for checkpoint_fname in os.listdir(args.save_dir):
            checkpoint_path = Path(args.save_dir) / checkpoint_fname
            if str.endswith(str(checkpoint_path), "best.pt"):
                print(f"evaluating checkpoint file {checkpoint_path}")
                eval_probe(args, False, checkpoint_path, logger)


if __name__ == '__main__':
    main()
