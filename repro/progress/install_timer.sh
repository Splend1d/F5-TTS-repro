#!/bin/bash
# Run on a Babel LOGIN node (compute nodes have no systemd user session): bash repro/progress/install_timer.sh
set -e
D=~/.config/systemd/user; mkdir -p $D
cp /data/group_data/UTD-NAS/chanjanh/speechgen/F5-TTS/repro/progress/systemd/f5-progress.{service,timer} $D/
systemctl --user daemon-reload
systemctl --user enable --now f5-progress.timer
systemctl --user list-timers f5-progress.timer
