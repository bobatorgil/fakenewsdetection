# Multimodal Fake News Detection

This repository contains a cleaned and modular implementation of a multimodal fake news detection model using news text, images, generated or real comments, and emotion features.

## Project structure

```text
journal_code_refactor/
├── main.py                         # Training entry point
├── requirements.txt                 # Python dependencies
├── run_train.sh                     # Example training command
├── model/
│   ├── config.py                    # Command-line arguments
│   ├── dataset.py                   # Dataset, emotion keys, and feature loading
│   ├── model.py                     # BERT-ViT gated multimodal model
│   ├── train.py                     # Training, evaluation, metrics, and optimizer utilities
│   └── utils.py                     # Reproducibility and file utilities
└── preprocess/
    ├── common.py                    # Shared preprocessing utilities
    ├── comment_generator.py         # LLM-based comment generation
    ├── comment_emotion.py           # Comment emotion extraction
    └── emotion.py                   # News-text emotion extraction
```

## Model overview

The model uses BERT for news text and comment text, ViT for images, and separate emotion features for text, image, and comments. Text and image emotions are fused with cross-modal attention. Comment emotion features are represented by mean and standard deviation statistics. Comment and emotion branches are gated and added to the text-image base branch through residual fusion.

## Data requirements

The default paths follow the original experiment environment:

- `train_jsonl`, `dev_jsonl`, `test_jsonl`: Fakeddit split files with `id`, `text`, and `2_way_label` columns.
- `data_dir/images`: Image directory containing files named by sample id, such as `id.jpg` or `id.png`.
- `image_emotions_csv`: CSV with an `id` column and image emotion columns.
- `text_emotions_csv`: CSV with `id` and `emotion` columns.
- `comment_emotions_csv`: CSV with `news_id` and `emotion` columns.
- `comments_jsonl`: JSONL file containing either grouped comments per news item or one comment per line.

The positive class for precision, recall, and F1-score is fake news, where `2_way_label = 0`.

## Installation

```bash
pip install -r requirements.txt
```

## Training

```bash
python main.py \
  --train_jsonl /path/to/train.jsonl \
  --dev_jsonl /path/to/dev.jsonl \
  --test_jsonl /path/to/test.jsonl \
  --data_dir /path/to/fakeddit \
  --image_emotions_csv /path/to/image_emotions.csv \
  --text_emotions_csv /path/to/text_emotions.csv \
  --comment_emotions_csv /path/to/comment_emotions.csv \
  --comments_jsonl /path/to/comments.jsonl \
  --output_dir results_gated_fusion
```

Or edit and run:

```bash
bash run_train.sh
```

## Outputs

The training script writes the following files to `output_dir`:

- `best_model.pt`: Best checkpoint selected by validation accuracy.
- `train_history.json`: Epoch-level training and validation logs.
- `summary.json`: Test metrics and main hyperparameters.

## Notes for reproducibility

The code fixes random seeds through Python, NumPy, and PyTorch. The first `backbone_warmup_epochs` epochs train BERT and ViT together with the task-specific layers. After that, BERT and ViT are frozen, and only the task-specific layers are updated.

## Preprocessing scripts

The preprocessing scripts are executable modules and do not rely on hard-coded local paths. Provide input and output paths through command-line arguments.

Generate LLM comments:

```bash
python -m preprocess.comment_generator \
  --data_dir /home/iai4/Desktop/ohw/data/fakeddit \
  --splits train dev test \
  --input_suffix .jsonl \
  --output_suffix _llm_comments.jsonl \
  --model qwen2.5vl \
  --resume
```

Extract emotions from generated or real comments:

```bash
python -m preprocess.comment_emotion \
  --comments_jsonl /home/iai4/Desktop/ohw/data/fakeddit/llm_comments.jsonl \
  --output_csv /home/iai4/Desktop/ohw/data/fakeddit/llm_comments_bertweet_emotions.csv \
  --resume
```

Extract emotions from news text:

```bash
python -m preprocess.emotion \
  --input_path /home/iai4/Desktop/ohw/data/fakeddit/train.jsonl \
  --input_format jsonl \
  --id_field id \
  --text_field text \
  --output_csv /home/iai4/Desktop/ohw/data/fakeddit/text_bertweet_emotions.csv \
  --resume
```

For MMFakeBench-style files, use `--id_field image_path --label_field gt_answers`.
