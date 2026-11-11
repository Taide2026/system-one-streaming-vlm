#!/bin/sh
# frame AUC with paired bootstrap; baseline = mix + KL (floor_d3nano_c) 1x
nice uv run --frozen python results/spark/frame_auc_ucf290/20261109-153701/frame-auc.py results/spark/frames_ucf290_f8/20261105-180529/manifest.json results/spark/frame_auc_ucf290/20261109-153701/frame-auc.json \
  mix1x=results/spark/sr_ucf290_d3nano_ft/20261106-142617/floor_d3nano_c/draft/scores.jsonl mix2x=results/spark/sr_ucf290_d3nano_ft/20261106-142617/floor_d3nano_c/B/scores.jsonl \
  oracle=results/spark/oracle_split_ucf290/20261109-151315/oracle.jsonl negonly=results/spark/oracle_split_ucf290/20261109-151315/negonly.jsonl fov2_2x=results/spark/sr_ucf290_d3nano_ft/20261106-142617/fov2_d3nano_c/B/scores.jsonl \
  ff1x=results/spark/sr_ucf290_d3nano_ft/20261109-015000/fovfull2x_d3nano_c/draft/scores.jsonl ffodd=results/spark/sr_ucf290_d3nano_ft/20261109-093702/fovfull2x_d3nano_c/odd/scores.jsonl \
  ff2x=results/spark/sr_ucf290_d3nano_ft/20261109-015000/fovfull2x_d3nano_c/B/scores.jsonl
