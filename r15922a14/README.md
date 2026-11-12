# ADL HW1: User State Prediction

Student ID: `r15922a14`

## 1. Task and submitted model

This project classifies Chinese and English user utterances into 10 binary user-state labels, using the current utterance and dialogue history. The encoder is `FacebookAI/xlm-roberta-base` with a 10-label classification head.

The submitted checkpoint is the context-128, weighted-loss, model-seed-42 run. Its original training directory is `outputs/v02_ctx128_weighted`; the model archive was built from that run's `best_model`. The selected checkpoint is epoch 9, after 12 completed training epochs. The commands below reproduce this configuration in a separate output directory.

## 2. Environment

The required TA environment is Ubuntu 20.04, Python 3.10, 32 GB RAM, an RTX 3080 with 10 GB VRAM, and 20 GB available disk space. The handout permits the following packages and their dependencies:

| Package | Required version |
| --- | --- |
| torch | 2.11.0 |
| transformers | 4.50.0 |
| numpy | 2.2.6 |
| scikit-learn | 1.7.2 |
| evaluate | 0.4.6 |
| matplotlib | 3.10.9 |
| gdown | 6.4.0 |
| tqdm | 4.70.1 |
| pandas | 2.3.3 |
| datasets | 2.21.0 |
| accelerate | 0.34.2 |

The code directly uses PyTorch, Transformers, NumPy, scikit-learn, and Python's standard library for training, inference, and validation analysis. `gdown` is available for downloading a Google Drive archive. No package installation is performed by `download.sh` or `run.sh`.

The local compatibility environment was checked on Windows (`Windows-10-10.0.26200-SP0`) with Python 3.10.9, PyTorch 2.11.0+cu126, Transformers 4.50.0, CUDA 12.6, and an NVIDIA RTX 4090 with 23.99 GiB VRAM. Offline inference on all 500 public examples passed with a 9 GiB PyTorch memory cap, batch size 4, and FP16; loading and inference took 6.40 seconds, with peak allocated GPU memory 1.082 GiB and peak reserved memory 1.154 GiB. A synthetic 750-record input with 1,250 context IDs also passed in 6.72 seconds, testing a larger input and extra unused context entries.

The actual submission ZIP was also tested in Ubuntu 22.04.4 LTS under WSL2, with Python 3.10.12, PyTorch 2.11.0+cu126, Transformers 4.50.0, CUDA 12.6, and the RTX 4090. A fresh ZIP extraction and anonymous model download passed; `download.sh` took 13.742 seconds. The complete `run.sh` command, including checksum verification, model extraction, startup, and offline prediction of all 500 records, took 13.607 seconds. The prediction CSV passed its independent format validator, and all five context-joining tests passed. Linux and Windows prediction CSVs were byte-identical. The timings describe the tested local environments. The TA's exact Ubuntu 20.04 / RTX 3080 combination has not been tested, and no actual private test set was available.

For a new training environment, create and activate a Python 3.10 virtual environment, install a CUDA-compatible build of the permitted PyTorch version, and install the core dependencies:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.11.0
python -m pip install transformers==4.50.0 numpy==2.2.6 scikit-learn==1.7.2 gdown==6.4.0
python check_environment.py
```

Use the TA's already provisioned environment when grading. The setup commands above are for training, before the offline inference phase.

## 3. Submission files

The submission ZIP must contain a single top-level folder named `r15922a14`, with the following files:

```text
r15922a14/
  README.md
  report.pdf
  download.sh
  run.sh
  train.py
  inference.py
  common.py
  analyze_validation.py
  compare_runs.py
  validate_output.py
  check_environment.py
  smoke_test_classifier.py
  smoke_test_preprocessing.py
  smoke_test_data_joining.py
  experiments/
    baseline/                 # Historical baseline source scripts
    mbert/                    # Historical mBERT source scripts
  report_sources/             # Report-generation code, original report, numeric evidence
```

Include every additional script used for training, prediction, or report figures. Training/testing data, downloaded archives, model weights, virtual environments, caches, and generated data/model outputs must be excluded from the COOL submission ZIP. Sanitized training configurations and numeric metrics are retained in `report_sources/` as report evidence, excluding split IDs and dialogue text. The trained model and tokenizer are obtained separately by `download.sh`.

## 4. Prepare training data

Obtain the official `train.jsonl` from the course data distribution and keep it outside the submission folder. Each line is one JSON record containing `id`, `language` (`zh` or `en`), `utterance`, `context` (a list of `role`/`content` dialogue turns), and `labels` (a list of the official label names). `train.py` validates the records before training.

For the official 3,000 training records, the code produces 2,400 training records and 600 validation records with `--validation-ratio 0.20 --split-seed 42`. It approximately stratifies by language and each record's rarest label. Model initialization and shuffled batches use the independent `--model-seed 42`. The selected split IDs are saved in `split_ids.json`.

The initial pretrained encoder/tokenizer is downloaded from Hugging Face during training if it is not already cached. This step requires network access. The final inference checkpoint includes its tokenizer and can be loaded entirely offline.

## 5. Train the final configuration

Run from the directory containing `train.py`. Replace the training-data path with the path to the official file:

```bash
python3 train.py \
  --train-file "/path/to/train.jsonl" \
  --output-dir "outputs/reproduce_ctx128_weighted_s42" \
  --model-name "FacebookAI/xlm-roberta-base" \
  --max-length 512 \
  --validation-ratio 0.20 \
  --train-batch-size 8 \
  --eval-batch-size 8 \
  --gradient-accumulation 1 \
  --learning-rate 2e-5 \
  --encoder-learning-rate 2e-5 \
  --classifier-learning-rate 2e-5 \
  --weight-decay 0.01 \
  --warmup-ratio 0.10 \
  --max-grad-norm 1.0 \
  --epochs 12 \
  --precision fp16 \
  --num-workers 0 \
  --split-seed 42 \
  --model-seed 42 \
  --input-order utterance-first \
  --context-max-tokens 128 \
  --turn-aware-context \
  --explicit-role-markers \
  --use-pos-weight \
  --checkpoint-metric fixed_harmonic \
  --early-stopping-patience 3 \
  --early-stopping-min-delta 0.0001
```

The effective training batch size is 8. The maximum input length is 512 tokens including special tokens, and history has a maximum budget of 128 tokens. The current utterance comes first. History uses recent complete turns and role markers such as `history user:` and `history assistant:`; the current utterance is marked `current user:`. If the newest turn alone exceeds its budget, the code keeps its beginning and end. A current utterance that alone exceeds the model budget is similarly truncated at both ends.

The loss is `BCEWithLogitsLoss` with per-label positive weight `sqrt(N_negative / N_positive)`, computed from the training split only. AdamW uses weight decay 0.01 for matrix parameters and zero decay for biases and one-dimensional parameters. The scheduler has 10% linear warmup followed by linear decay; gradient norms are clipped to 1.0.

Checkpoint selection maximizes the harmonic mean of macro-F1 and micro-F1 at a fixed threshold of 0.5. Training stops after three epochs without an improvement exceeding 0.0001, or after 12 epochs. Once selection finishes, the best checkpoint is reloaded and thresholds are calibrated once on the validation split. Calibration searches a global threshold from 0.10 to 0.90 in steps of 0.05, then performs one coordinate-descent pass over the 10 label thresholds to maximize the same harmonic mean.

Training writes:

```text
outputs/reproduce_ctx128_weighted_s42/
  split_ids.json
  metrics_history.json
  run_summary.json
  best_model/
    config.json
    pytorch_model.bin
    tokenizer.json
    tokenizer_config.json
    special_tokens_map.json
    labels.json
    preprocess_config.json
    thresholds.json
    validation_metrics.json
```

The recorded submission run used the same arguments, with `outputs/v02_ctx128_weighted` as its output directory. Use the pinned package versions for reproduction; changing hardware or software can affect floating-point results.

## 6. Reproduce validation analysis and comparisons

Analyze the saved best checkpoint and its exact saved validation IDs:

```bash
python3 analyze_validation.py \
  --train-file "/path/to/train.jsonl" \
  --run-dir "outputs/reproduce_ctx128_weighted_s42" \
  --batch-size 8 \
  --precision fp16
```

This writes `analysis/diagnostics.json`, `analysis/per_label_metrics.csv`, and `analysis/error_examples.json` under the run directory. It reports per-label precision/recall/F1, exact match, Hamming loss, language groups, input-length groups, and context/truncation groups. It reads the official training file and saved validation split, without requiring test labels.

For an XLM-R ablation, repeat the training command with a different output directory and the relevant changed arguments. Keep all other arguments from Section 5:

| Experiment | Argument changes from the final configuration |
| --- | --- |
| No history | `--context-max-tokens 0` |
| Larger history budgets | `--context-max-tokens 256` or `512` |
| Unweighted loss | Omit `--use-pos-weight` |
| Other model seeds | `--model-seed 17` or `87`, retaining `--split-seed 42` |
| Legacy context packing | `--input-order context-first --context-max-tokens -1 --no-turn-aware-context --no-explicit-role-markers` |

The legacy row reproduces `v02_legacy_weighted`: it retains weighted loss, 12 maximum epochs, and the same fixed-threshold checkpoint-selection rule. Run validation analysis for each run, then compare them:

```bash
python3 compare_runs.py \
  outputs/reproduce_ctx128_weighted_s42 \
  /path/to/another/completed/run
```

For the mBERT comparison, use the preserved scripts under `experiments/mbert/`. Their preprocessing emits BERT's segment IDs when required by its tokenizer. From the submission root, run:

```bash
python3 experiments/mbert/train.py \
  --train-file "/path/to/train.jsonl" \
  --output-dir "outputs/reproduce_mbert_ctx128_weighted_s42" \
  --model-name "google-bert/bert-base-multilingual-cased" \
  --max-length 512 \
  --validation-ratio 0.20 \
  --train-batch-size 8 \
  --eval-batch-size 8 \
  --gradient-accumulation 1 \
  --learning-rate 2e-5 \
  --weight-decay 0.01 \
  --warmup-ratio 0.10 \
  --max-grad-norm 1.0 \
  --epochs 12 \
  --precision fp16 \
  --num-workers 0 \
  --split-seed 42 \
  --model-seed 42 \
  --input-order utterance-first \
  --context-max-tokens 128 \
  --turn-aware-context \
  --explicit-role-markers \
  --use-pos-weight \
  --checkpoint-metric fixed_harmonic \
  --early-stopping-patience 3 \
  --early-stopping-min-delta 0.0001

python3 experiments/mbert/analyze_validation.py \
  --train-file "/path/to/train.jsonl" \
  --run-dir "outputs/reproduce_mbert_ctx128_weighted_s42" \
  --batch-size 8 \
  --precision fp16
```

For the three early baseline rows, use the original code in `experiments/baseline/`. Its default model, length, optimizer, data split, and batch-size arguments are those shown in Section 5, while `--seed 42` sets its random seed. Reproduce the different training schedules as follows:

```bash
python3 experiments/baseline/train.py \
  --train-file "/path/to/train.jsonl" \
  --output-dir "outputs/reproduce_xlmr_baseline" \
  --epochs 4 --gradient-accumulation 4 --seed 42 --precision fp16

python3 experiments/baseline/train.py \
  --train-file "/path/to/train.jsonl" \
  --output-dir "outputs/reproduce_xlmr_more_steps" \
  --epochs 8 --gradient-accumulation 1 --seed 42 --precision fp16

python3 experiments/baseline/train.py \
  --train-file "/path/to/train.jsonl" \
  --output-dir "outputs/reproduce_xlmr_weighted" \
  --epochs 8 --gradient-accumulation 1 --seed 42 --precision fp16 \
  --use-pos-weight
```

Use `experiments/baseline/analyze_validation.py` with the corresponding run directory to reproduce baseline diagnostics. The archived baseline performs checkpoint selection and threshold tuning differently from V2; use its own source scripts for those historical results.

The recorded checkpoint's calibrated validation results are:

| Metric | Score |
| --- | ---: |
| Macro-F1 | 0.835055 |
| Micro-F1 | 0.843026 |
| Harmonic mean of macro-F1 and micro-F1 | 0.839021 |
| Exact match accuracy | 0.593333 |
| Hamming loss | 0.059833 |

Before threshold calibration, epoch 9 has macro-F1 0.827266, micro-F1 0.837088, and harmonic mean 0.832148. These are validation results on 600 records. Checkpoint selection and threshold calibration share this split, so the calibrated scores are not an independent test estimate.

## 7. Download the trained model

`download.sh` uses the fixed Google Drive file ID `1Hb56bCL1iYuIblffYXAyM5bTj81yRqqj` with the permitted `gdown==6.4.0` and `--no-cookies`. The complete archive was downloaded anonymously from a clean directory on 2026-10-04. Its byte count and SHA-256 match the verified local model archive exactly. The download link is https://drive.google.com/file/d/1Hb56bCL1iYuIblffYXAyM5bTj81yRqqj/view .

Run the download step before any other submitted script:

```bash
bash ./download.sh
```

The download artifact is `model.tar.gz`, containing the trained checkpoint, tokenizer, preprocessing configuration, labels, and thresholds under `model/`. The archive is 803,048,033 bytes (about 765.85 MiB), below the 4 GB download limit. Its verified SHA-256 is:

```text
06354df9b2e11b5f0c09326f4699ce106395440128304d26b74890310343e5d0
```

`download.sh` is restricted to downloading; archive verification/extraction and inference occur in `run.sh`. The TA supplies `context.json` and the public/private `test.json` through arguments, so those inputs are not embedded in the model archive. The official `train.jsonl` is needed only for retraining, as described above.

The complete anonymous download took 14.30 seconds on the local connection, including Python startup. This measurement uses the local bandwidth. At a sustained 10 MB/s, transferring 803 MB takes about 80 seconds plus connection overhead. The remote archive must remain accessible at least through 2026-10-18 23:59 (Asia/Taipei) and must not be modified after the 2026-10-04 23:59 deadline.

## 8. Offline inference

Use exactly the required three-argument interface:

```bash
bash ./run.sh \
  /path/to/context.json \
  /path/to/test.json \
  /path/to/pred/prediction.csv
```

The arguments are `$1` = context-file path, `$2` = test-file path, and `$3` = output CSV path. `run.sh` loads `model/` beside the script (extracting the downloaded archive as needed), uses batch size 4 and CUDA FP16 inference, and enables offline Hugging Face settings. Both model and tokenizer loading use `local_files_only=True`. On a machine without CUDA, the Python inference code uses FP32 on CPU.

The output has one row per supplied test record, preserving its ID order. The header is:

```csv
id,Ask_Price,Ask_Service,Ask_Spec,Compare_Competitor,Confirm_Order,Doubt,Need_More_Evidence,Need_Time_To_Think,State_Need,Too_Expensive
```

All label values are integers 0 or 1. The row count is determined by the supplied test file, so the code supports both public and private test sets. The two-hour requirement applies to inference on both test sets; an RTX 4090 local run does not by itself establish the runtime on the TA's RTX 3080.

## 9. Validate the prediction format

```bash
python3 validate_output.py \
  --test-file "/path/to/test.json" \
  --prediction-file "/path/to/pred/prediction.csv"
```

For the 500-record public test set, successful validation prints:

```text
OK: 500 rows, exact header/order, binary values only.
```

For a differently sized private test set, the count changes to match that input. The inference program also validates its output automatically.

Check that test examples join their contexts by ID, including reordered input and extra context entries, with:

```bash
python3 smoke_test_data_joining.py
```

This runs five tests without model loading or network access. It also checks that missing contexts and duplicate test IDs are rejected and empty dialogue histories are accepted.

## 10. Report sources

`report.pdf` preserves the original eight-page report and adds a clarification appendix. The original report is included as `report_sources/original_report.pdf`. Numeric evidence is preserved under `report_sources/evidence/`; it contains aggregate results and training configurations, without example texts, labels for individual examples, split IDs, model weights, or tokenizer files. The original report's terminal-output figures can be reproduced with the training, `compare_runs.py`, and `analyze_validation.py` commands above.

To rebuild the PDF appendix in a separate document-authoring environment:

```bash
python -m pip install pypdf reportlab
python report_sources/build_report.py \
  --source report_sources/original_report.pdf \
  --output report.pdf
```

These two optional PDF-authoring dependencies are never imported by training or inference and are not required in the TA grading environment. The archived `experiment_sources/original_run_queue.py` preserves the original local experiment orchestration source; use the explicit portable training commands above when reproducing experiments.

`verification/` contains aggregate local verification records and the optional Linux test and memory-profiling helpers. These helpers are separate from the required `download.sh` and `run.sh` grading entry points. The Linux helper accepts a directory containing `r15922a14.zip`, `context.json`, and `test.json` as its first argument; it extracts the ZIP into a fresh directory, runs the download/inference interface, and validates the CSV.
