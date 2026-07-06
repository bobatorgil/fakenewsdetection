#!/usr/bin/env bash
set -euo pipefail

python main.py \
  --train_jsonl /home/iai4/Desktop/ohw/data/fakeddit/real_comments_5_filtered_splits/train_with_emotion_real_comments_5.jsonl \
  --dev_jsonl /home/iai4/Desktop/ohw/data/fakeddit/real_comments_5_filtered_splits/dev_with_emotion_real_comments_5.jsonl \
  --test_jsonl /home/iai4/Desktop/ohw/data/fakeddit/real_comments_5_filtered_splits/test_with_emotion_real_comments_5.jsonl \
  --data_dir /home/iai4/Desktop/ohw/data/fakeddit \
  --image_emotions_csv /home/iai4/Desktop/ohw/data/fakeddit/emotionclip_imagewise_results.csv \
  --text_emotions_csv /home/iai4/Desktop/ohw/data/fakeddit/fakeddit_bertweet_emotions.csv \
  --comment_emotions_csv /home/iai4/Desktop/ohw/data/fakeddit/llm_comments_bertweet_emotions.csv \
  --comments_jsonl /home/iai4/Desktop/ohw/data/fakeddit/llm_comments.jsonl \
  --output_dir results_gated_fusion
