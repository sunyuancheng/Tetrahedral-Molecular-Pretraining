ulimit -c unlimited

[ -z "${layers}" ] && layers=12
[ -z "${hidden_size}" ] && hidden_size=768
[ -z "${ffn_size}" ] && ffn_size=768
[ -z "${num_head}" ] && num_head=32
[ -z "${batch_size}" ] && batch_size=256
[ -z "${update_freq}" ] && update_freq=1
[ -z "${seed}" ] && seed=1
[ -z "${clip_norm}" ] && clip_norm=5
[ -z "${data_path}" ] && data_path='./'
[ -z "${save_path}" ] && save_path='./logs/finetune_lba/exp-dataset-ligand_binding_affinity-split_60-lr-2e-5-end_lr-1e-9-tsteps-60000-wsteps-6000-L12-D768-F768-H32-SLN-false-BS32-CLIP5-dp0.0-attn_dp0.1-wd0.0-dpp0.0/SEED1-TASK-LOSS-L1-STD-std_logits-RF-'
[ -z "${dropout}" ] && dropout=0.0
[ -z "${act_dropout}" ] && act_dropout=0.1
[ -z "${attn_dropout}" ] && attn_dropout=0.1
[ -z "${weight_decay}" ] && weight_decay=0.0
[ -z "${sandwich_ln}" ] && sandwich_ln="false"
[ -z "${droppath_prob}" ] && droppath_prob=0.1
[ -z "${mode_prob}" ] && mode_prob="1.0,0.0,0.0"

[ -z "${dataset_name}" ] && dataset_name="ligand_binding_affinity-split_60"
[ -z "${add_3d}" ] && add_3d="true"
[ -z "${no_2d}" ] && no_2d="true"
[ -z "${num_3d_bias_kernel}" ] && num_3d_bias_kernel=128

python evaluate_lba.py \
	--user-dir $(realpath ./Transformer-M) \
	--data-path $data_path \
	--num-workers 16 --ddp-backend=legacy_ddp \
	--dataset-name $dataset_name \
	--batch-size $batch_size --data-buffer-size 20 \
	--task lba_finetune --criterion lba_finetune --arch transformer_m --num-classes 1 \
	--remove-head \
	--encoder-layers $layers --encoder-attention-heads $num_head --add-3d --no-2d --num-3d-bias-kernel $num_3d_bias_kernel \
	--encoder-embed-dim $hidden_size --encoder-ffn-embed-dim $ffn_size --droppath-prob $droppath_prob \
	--attention-dropout $attn_dropout --act-dropout $act_dropout --dropout $dropout \
	--save-dir $save_path --noise-scale 0.0 --mode-prob $mode_prob --split test --metric rmse
