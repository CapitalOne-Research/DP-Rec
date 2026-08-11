## 1. Create Environment

```bash
conda env create -f train.yml
conda activate dprec_train
```

> Installs the CUDA build of PyTorch (`torch==2.12.1+cu130`) from the internal
> conda/pip channels. Tested with PyTorch 2.12 + CUDA 13.0.

---

## 2. Data Download

| Dataset | Download |
|---------|----------|
| ML-1M | https://files.grouplens.org/datasets/movielens/ml-1m.zip |
| ML-10M | https://files.grouplens.org/datasets/movielens/ml-10m.zip |
| KuaiRand-27K | https://zenodo.org/records/10439422/files/KuaiRand-27K.tar.gz |

Download and extract into a local directory, e.g. `~/datasets/`.

```bash
# ML-1M
curl -L -O https://files.grouplens.org/datasets/movielens/ml-1m.zip
unzip ml-1m.zip

# ML-10M
curl -L -O https://files.grouplens.org/datasets/movielens/ml-10m.zip
unzip ml-10m.zip

# KuaiRand-27K
curl -L -O https://zenodo.org/records/10439422/files/KuaiRand-27K.tar.gz
tar -xzf KuaiRand-27K.tar.gz
```

---

## 3. Data Pre-process

Processed files are written to the `data/` directory as `<dataset>.txt` with format `user item timestamp` per line.

**MovieLens (ML-1M and ML-10M)**

```bash
python src/data_process_mlm.py ml-1m/ratings.dat
python src/data_process_mlm.py ml-10M100K/ratings.dat --dataset-name ml-10m
```

> `ml-10m.zip` extracts to a folder named `ml-10M100K/`, so pass
> `--dataset-name ml-10m` to keep the training convention (`dataset=ml-10m`).

Users with fewer than `--min-seq-len` interactions are dropped (default: 5). To
build a long-sequence variant (e.g. ML-10M-L), raise the threshold:

```bash
python src/data_process_mlm.py ml-10M100K/ratings.dat --dataset-name ml-10m --min-seq-len 500
```

**KuaiRand-27K**

```bash
# Point at the KuaiRand-27K data directory — the log_standard_* parts are
# merged automatically (the log_random_* exposure log is excluded).
python src/data_process_kuairec.py KuaiRand-27K/data

# Or pass a single pre-merged CSV
python src/data_process_kuairec.py KuaiRand-27K/data/full_data.csv
```

---

## 4. Train DP-Rec

DP-Rec requires a pre-trained patcher (boundary detector) checkpoint. Train it first.

All training is launched via `main.py` using [Hydra](https://hydra.cc/) configuration. Model configs live in `configs/model/`.

**Step 1 — Train the patcher**

```bash
python main.py \
  model=dp_rec_patcher \
  experiment_folder=my_experiment \
  model_suffix=patcher \
  dataset=ml-1m \
  local_data_dir=data/
```

The checkpoint is saved to `exp/my_experiment/dp_rec/dp_rec_patcher_ml-1m_<hash>/`.

**Step 2 — Train DP-Rec**

```bash
python main.py \
  model=dp_rec \
  experiment_folder=my_experiment \
  model_suffix=dprec \
  dataset=ml-1m \
  local_data_dir=data/ \
  patcher_model_path=exp/my_experiment/dp_rec/dp_rec_patcher_ml-1m_<hash>
```

### Overriding config parameters

Any parameter can be overridden on the command line with `++`:

```bash
python main.py \
  model=dp_rec \
  experiment_folder=ablation \
  model_suffix=maxlen50 \
  dataset=ml-1m \
  local_data_dir=data/ \
  patcher_model_path=exp/my_experiment/dp_rec/dp_rec_patcher_ml-1m_<hash> \
  ++model_args.maxlen=50 \
  ++training_args.num_epochs=200
```

---

## 5. Evaluate Model

Evaluation runs automatically at the end of training. To run inference on a saved checkpoint without retraining:

```bash
python main.py \
  dataset=ml-1m \
  local_data_dir=data/ \
  task=inference_only \
  model_dir=exp/my_experiment/dp_rec/dp_rec_dprec_ml-1m_<hash>
```

Results are written to `results.csv` inside the experiment directory. Each row corresponds to one evaluation epoch with columns such as `Valid_NDCG@10`, `Test_NDCG@10`, `Test_HR@10`, etc.

---

## Configuration Reference

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model` | Config file in `configs/model/` | `dp_rec` |
| `experiment_folder` | Top-level directory grouping related runs | required |
| `model_suffix` | Suffix identifying this run | required |
| `dataset` | Dataset name, must match a `.txt` file in `local_data_dir` | required |
| `local_data_dir` | Path to processed data files | `../data/` |
| `local_output_dir` | Path where checkpoints and logs are saved | `exp/${experiment_folder}/` |
| `task` | `train` or `inference_only` | `train` |
| `model_dir` | Path to saved checkpoint (required for `inference_only`) | `null` |
| `patcher_model_path` | Path to pre-trained patcher checkpoint (required for `dp_rec` training) | required |
| `training_args.num_epochs` | Number of training epochs | model-specific |
| `model_args.maxlen` | Maximum sequence length | model-specific |
| `training_args.batch_size` | Batch size | model-specific |
| `training_args.lr` | Learning rate | model-specific |

### Output files per run

Each run produces a directory under `local_output_dir` containing:

- `config.yaml` — full Hydra config used for the run
- `logs.txt` — training logs
- `results.csv` — per-epoch validation and test metrics
- `state_dict.pth` — best model checkpoint
- `params.json` — model architecture metadata (used by patcher loading)
