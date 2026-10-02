#!/bin/bash
# Stop a self-resubmitting run cleanly: bash repro/stop_train.sh [TAG]   (resume later: rm repro/state/TAG.stop && bash repro/submit_train.sh)
TAG=${1:-main}; F5=/data/group_data/UTD-NAS/chanjanh/speechgen/F5-TTS
touch $F5/repro/state/$TAG.stop
scancel -u $USER -n f5-train-$TAG && echo "stopped f5-train-$TAG (stop file set)"
