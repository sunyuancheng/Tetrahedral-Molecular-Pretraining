import torch
import numpy as np
from fairseq import checkpoint_utils, utils, options, tasks
from fairseq.logging import progress_bar
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
import ogb
import sys
import os
from pathlib import Path
from sklearn.metrics import roc_auc_score

import sys
from os import path

sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

import logging

def eval(args, use_pretrained, checkpoint_path=None, logger=None):
    cfg = convert_namespace_to_omegaconf(args)
    np.random.seed(cfg.common.seed)
    utils.set_torch_seed(cfg.common.seed)

    # initialize task
    task = tasks.setup_task(cfg.task)
    model = task.build_model(cfg.model)

    # load checkpoint

    model_state = torch.load(checkpoint_path)["model"]
    model.load_state_dict(
        model_state, strict=True, model_cfg=cfg.model
    )
    del model_state

    model.to(torch.cuda.current_device())
    # load dataset
    import pdb; pdb.set_trace()
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
            y = model(**sample["net_input"])[0][:, 0, :]
            y_pred.extend(y.detach().cpu())
            y_true.extend(sample["target"].detach().cpu().reshape(-1)[:y.shape[0]])
            torch.cuda.empty_cache()

    # save predictions
    y_pred = torch.Tensor(y_pred)
    y_true = torch.Tensor(y_true)
    import pdb; pdb.set_trace()

    if "30" in cfg.task.dataset_name:
        train_mean = 6.5239
        train_std = 2.0012
    elif "60" in cfg.task.dataset_name:
        train_mean = 6.4793
        train_std = 1.9701
    y_pred = y_pred * train_std + train_mean

    # evaluate pretrained models

    from scipy.stats import spearmanr, pearsonr
    rmse = torch.sqrt(torch.sum((y_pred - y_true)**2))
    spearman = pearsonr(y_pred.squeeze(-1).cpu().numpy(), y_true.squeeze(-1).detach().cpu().numpy())[0]
    eval_pearson = spearmanr(y_pred.squeeze(-1).cpu().numpy(), y_true.squeeze(-1).detach().cpu().numpy())[0]
    logger.info(f"rmse: {rmse}")
    logger.info(f"spearman: {spearman}")
    logger.info(f"eval_pearson: {eval_pearson}")



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
    for checkpoint_fname in os.listdir(args.save_dir):
        checkpoint_path = Path(args.save_dir) / checkpoint_fname
        # if str(checkpoint_path)[-3:] == '.pt':
        if str(checkpoint_path).endswith('last.pt'):
            logger.info(f"evaluating checkpoint file {checkpoint_path}")
            result = eval(args, False, checkpoint_path, logger)
            # open(str(checkpoint_path)[:-3] + f'_{args.split}_{result:.5f}.txt', 'w')


if __name__ == '__main__':
    main()
