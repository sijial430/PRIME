#!/bin/bash

for script in stage1_filter.py stage2_format_choice.py stage3_merge.py stage4_judge.py; do
    python "$script"
done