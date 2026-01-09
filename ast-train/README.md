# AST Train

Cleaned training/evaluation entrypoint for the AST model used in this repo.

## Expected data layout
By default, this folder reads JSON metadata from `../data`:

- `../data/train_data.json`
- `../data/valid_data.json`
- `../data/test_data.json`

Each JSON file should contain:

```
{
  "data": [
    {"wav": "/path/to/audio.wav", "labels": 0, "seg_start": 1.2, "seg_end": 4.8},
    {"wav": "/path/to/audio.wav", "labels": 1}
  ]
}
```

Optional fields: `seg_start`, `seg_end` (seconds). Labels are integers (0/1).

CSV is also supported with columns:
- `filepath` or `wav` or `path`
- `label` or `labels` or `target`
- optional `seg_start`, `seg_end`

## Usage

Train and evaluate with defaults:

```
python run.py
```

Override data paths:

```
python run.py --data_root /path/to/data
```

Leave-one-model-out split:

```
python run.py --leave_model <model_name>
```

## Outputs

Outputs are written under `experiments/` in this folder:
- `experiments/default/models/best_audio_model.pth`
- `experiments/default/eval_result.csv`
- `experiments/default/test_predictions.csv`
- `experiments/default/roc_ast.csv`

## Notes

- Non-WAV inputs require torchaudio with an ffmpeg backend; ensure ffmpeg is installed.
- `timm==0.4.5` is required to match the original AST implementation.
