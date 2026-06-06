#!/bin/bash

#~/anaconda3/envs/offrl/bin/python main_off.py --multirun trainer=iql env.name='hopper-medium-replay-v2' \
#device.seed=0,1,2,3 device.gpu_idx=0

~/anaconda3/envs/offrl/bin/python main_off.py --multirun trainer=iql spec=iql_h_m_e_v2 env.name='hopper-medium-expert-v2' \
device.seed=0,1,2,3 device.gpu_idx=0





#~/anaconda3/envs/offrl/bin/python main_off.py --multirun trainer=iql env.name='antmaze-umaze-v0' device.seed=0,1 \
#device.gpu_idx=0 rlalg.num_eval_steps_per_epoch=70000 rlalg.num_epochs_per_log_interval=10 rlalg.start_epoch=-100 \
#trainer.trainer_kwargs.quantile=0.9 trainer.trainer_kwargs.beta=10.0 trainer.trainer_kwargs.total_training_steps=1E5 \
#trainer.reward_norm=False

