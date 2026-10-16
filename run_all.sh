#!/bin/bash
# Sequential, restartable pipeline for every GHOST v2 result (CPU, 2 threads).
# Each step is skipped when its result file already exists.
set -x
cd "$(dirname "$0")"
export NT=2
step() { out=$1; shift; [ -f "results/$out" ] && return 0; "$@"; }
while pgrep -f "extract_mia.py resnet20 extract" > /dev/null || pgrep -f "exact_overhead.py resnet20" > /dev/null; do sleep 30; done
step victim_resnet20_s0.json python3 train_victim.py resnet20 30 0 >> results/nohup_r20.txt 2>&1
step exact_resnet20.json     python3 exact_overhead.py resnet20  > results/log_exact_r20.txt 2>&1
step token_tests.json        python3 token_tests.py              > results/log_token.txt 2>&1
step mia_resnet20.json       python3 extract_mia.py resnet20 mia > results/log_mia_r20.txt 2>&1
step victim_resnet20_s1.json python3 train_victim.py resnet20 30 1 >> results/nohup_r20s1.txt 2>&1
for c in scratch plain ghost arrow_own unmix_own arrow_oracle unmix_oracle projK_oracle ps_arrow_oracle left_projK_oracle; do
  STEPS=1000 step steal_resnet20_n5000_$c.json python3 steal.py resnet20 5000 $c >> results/log_steal_r20_5000.txt 2>&1; done
step victimft_resnet20.json python3 finetune_public.py > results/log_ft.txt 2>&1
for c in public plain ghost ps_arrow_oracle left_projK_oracle arrow_oracle unmix_oracle projK_oracle; do
  VICTIM=resnet20_victimft ORACLE=resnet20_victim_s1 TAG=pub BASE=ref STEPS=1000 step stealpub_resnet20_n5000_$c.json python3 steal.py resnet20 5000 $c >> results/log_steal_pub.txt 2>&1; done
VICTIM=resnet20_victimft TAG=pub step blind_icapub_resnet20_L0.json python3 blind_ica.py resnet20 0 > results/log_ica0.txt 2>&1
VICTIM=resnet20_victimft TAG=pub step blind_icapub_resnet20_L3.json python3 blind_ica.py resnet20 3 > results/log_ica3.txt 2>&1
STEPS=1000 step inversion_resnet20.json python3 inversion.py resnet20 > results/log_inv_r20.txt 2>&1
STEPS=1000 step extract_resnet20.json python3 extract_mia.py resnet20 extract > results/log_extract_r20.txt 2>&1
for n in 1000 10000; do for c in scratch plain ghost unmix_oracle projK_oracle; do
  STEPS=1000 step steal_resnet20_n${n}_$c.json python3 steal.py resnet20 $n $c >> results/log_steal_r20_$n.txt 2>&1; done; done
step victim_vgg11h_s0.json python3 train_victim.py vgg11h 30 0 >> results/nohup_vgg.txt 2>&1
step exact_vgg11h.json python3 exact_overhead.py vgg11h > results/log_exact_vgg.txt 2>&1
step mia_vgg11h.json python3 extract_mia.py vgg11h mia > results/log_mia_vgg.txt 2>&1
for c in scratch plain ghost arrow_own unmix_own; do
  STEPS=1000 step steal_vgg11h_n5000_$c.json python3 steal.py vgg11h 5000 $c >> results/log_steal_vgg_5000.txt 2>&1; done
echo ALLDONE > results/ALLDONE
