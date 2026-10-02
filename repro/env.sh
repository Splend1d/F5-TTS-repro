# Common env for F5-TTS reproduction. All caches on UTD-NAS (home & user_data are full).
export R=/data/group_data/UTD-NAS/chanjanh/speechgen
export F5=$R/F5-TTS
export HF_TOKEN=$(cat /data/user_data/chanjanh/.cache/huggingface/token)
export HF_HOME=$R/cache/hf_home
export HF_HUB_CACHE=$R/cache/hf_hub
export HF_DATASETS_CACHE=$R/cache/hf_datasets
export UV_CACHE_DIR=$R/cache/uv
export TORCH_HOME=$R/cache/torch
export XDG_CACHE_HOME=$R/cache/xdg
export MODELSCOPE_CACHE=$R/cache/modelscope
export WANDB_DIR=$R/cache/wandb
# A queued Slurm job reads this file when it starts, so run-specific settings can
# be changed without cancelling the job and losing its accrued queue priority.
F5_RUN_ENV=$F5/repro/state/${TAG:-main}.env
[ -r "$F5_RUN_ENV" ] && source "$F5_RUN_ENV"
unset F5_RUN_ENV
export WANDB_MODE=${WANDB_MODE:-offline}
export HF_HUB_ENABLE_HF_TRANSFER=0
[ -f $F5/.venv/bin/activate ] && source $F5/.venv/bin/activate
cd $F5
