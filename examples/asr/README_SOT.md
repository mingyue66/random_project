# SOT (Serialized Output Training) for ASR

Train ASR models to predict speaker change tokens (`<sc>`) along with transcription.

## Files

### Related Files
- `train_sot.py` - SOT training script with HF tokenizer support
- `trainer_sot.py` - Trainer with `<sc>` token monitoring
- `configs/train_sot.yaml` - SOT training configuration
- `data_module.py` - Added `special_tokens_to_keep=["<sc>"]` to protect `<sc>` during text normalization

## Quick Start

```bash
# Training from scratch
torchrun --nproc_per_node=1 train_sot.py \
  --config-name train_sot \
  exp_dir=exp/sot_alimeeting \
  tokenizer=bert-base-chinese \
  data.train_data_config=configs/data_configs/train_data_config.yaml \
  data.valid_data_config=configs/data_configs/valid_data_config.yaml \
  sot_training=true
```


## Data Requirements

Training data must contain `<sc>` tokens in the reference text to mark speaker changes:

```json
{
  "supervisions": [
    {
      "text": "hello <sc> world <sc> how are you"
    }
  ]
}
```

## How It Works

### 1. Tokenizer Setup
- Uses HuggingFace `AutoTokenizer` (e.g., `bert-base-chinese`)
- Adds `<sc>` as a special token via `add_special_tokens()`
- `<sc>` gets assigned the next token ID (e.g., 21128)

### 2. Text Normalization Protection
In `data_module.py`, `<sc>` is protected during text normalization:
```python
special_tokens_to_keep=["<sc>"]
```
This prevents `<sc>` from being removed by `remove_in_brackets=True`.

### 3. Model Training
- Model vocab expands to include `<sc>` token
- Loss includes `<sc>` as a prediction target
- Model learns when to output `<sc>` (at speaker boundaries)

### 4. Monitoring
During training, logs show `<sc>` token statistics:
```
[SOT Batch 1] Found 24/26 utterances with <sc> token
  Example 0: 11 <sc> token(s) found
  Text preview: 'hello <sc> world <sc> how are you...'
  
[SOT Stats @ step 100] Batches with <sc>: 85/100 (85.0%), Total <sc> utterances: 1234
```


## Notes

- If training data doesn't contain `<sc>`, logs will show `0/N (0.0%)` batches with `<sc>`
- Model will train normally but won't learn to predict speaker changes
- `<sc>` token embedding exists but remains untrained
