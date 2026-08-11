# #Ref: https://github.com/nancheng58/RecMamba/tree/main

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import time
import torch
# torch.set_default_dtype(torch.bfloat16)
import hydra
from omegaconf import DictConfig, OmegaConf
import argparse

from src.models import SASRec, LONGER, HSTU
from src.gru4rec_official import GRU4RecWrapper

# DP-Rec uses a dash in the directory name which is not a valid Python identifier.
# We register it under the alias DP_Rec so relative imports inside the package work.
import importlib.util as _ilu, sys as _sys
_dp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DP-Rec")
_dp_name = "DP_Rec"
if _dp_name not in _sys.modules:
    _subnames = [
        "ops",
        "behavioral_boundary_detector", "simple_patchers", "temporal_local_encoder",
        "temporal_latent_transformer", "local_decoder", "dprec",
    ]
    # Register package stub first so submodule relative imports resolve
    import types as _types
    _pkg = _types.ModuleType(_dp_name)
    _pkg.__path__ = [_dp_dir]
    _pkg.__package__ = _dp_name
    _sys.modules[_dp_name] = _pkg
    for _s in _subnames:
        _sp = _ilu.spec_from_file_location(f"{_dp_name}.{_s}", os.path.join(_dp_dir, f"{_s}.py"))
        _m = _ilu.module_from_spec(_sp)
        _m.__package__ = _dp_name
        _sys.modules[f"{_dp_name}.{_s}"] = _m
        _sp.loader.exec_module(_m)
    _init_sp = _ilu.spec_from_file_location(_dp_name, os.path.join(_dp_dir, "__init__.py"),
                                             submodule_search_locations=[_dp_dir])
    _init_sp.loader.exec_module(_pkg)
from DP_Rec import DPREC, BehavioralBoundaryModel

from src.utils import *
from flops import compute_flops
from tqdm import tqdm
#from torchinfo import summary
import json
import csv
import os
            
def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

def count_parameters(model, include_embeddings = False):
    if include_embeddings:
        eval_params = sum(p.numel() for p in model.parameters())
        train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return train_params, eval_params
    embedding_params = set()
    for m in model.modules():
        if isinstance(m, torch.nn.Embedding):
            for p in m.parameters():
                embedding_params.add(p)
    eval_params = sum(p.numel() for p in model.parameters() if p not in embedding_params)
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad and p not in embedding_params)
    return train_params, eval_params


def write_metrics(logger, loss, epoch, T, val_metrics, test_metrics, model, fname, m_size, d_size, flops=None):
    """
    Write metrics to CSV file

    Args:
        results_path: Path to the CSV file
        metrics_data: Dictionary containing all metrics to write
    """

    val_metrics = {f'Valid_{k}': f'{v:.4f}' for k, v in val_metrics.items()}
    test_metrics = {f'Test_{k}': f'{v:.4f}' for k, v in test_metrics.items()}

    result = {**val_metrics, **test_metrics}
    result['Best Epoch'] = str(epoch)
    result['loss'] = f'{loss:.6f}'
    result['Time(s)'] = f'{T:.2f}(s)'
    result['m_size'] = f'{m_size:.2f}(MB)' # TODO move this to the configuration later
    result['d_size'] = f'{d_size:.2f}(MB)'

    if flops:
        result['mean_flops']  = f"{flops['mean_flops']:.0f}"
        result['p99_flops']   = f"{flops['p99_flops']:.0f}"
        result['total_flops'] = f"{flops['total_flops']:.0f}"

    logger.info(result)
    
    write_header = not os.path.isfile(fname)

    with open(fname, mode='a', newline='') as f:
        writer = csv.writer(f)
        
        # Write header if file doesn't exist
        if write_header:
            writer.writerow(result.keys())
        
        writer.writerow(result.values())



def train(args, dataset, experiment_path, logger):

    if args.model_args.get('train_patcher', False) == True:
        fname = 'patcher.pth'

    # load dataset
    # ---------------------------------
    [user_train, user_valid, user_test, usernum, itemnum, timenum] = dataset
    if args.model_args.backbone == 'tisas':
        relation_matrix = Relation(user_train, usernum, args.model_args.maxlen, args.model_args.get('timespan', 256), args.model_args.get('sliding_window'))
    else:
        relation_matrix = None
        logger.info("Skipping relation matrix computation (not needed for this model)")

    logger.info(f"Total train sequences: {len(user_train)}")
    num_batch = len(user_train) // args.training_args.batch_size
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    logger.info(f'Average sequence length: {cc / len(user_train):.2f}')
    logger.info(f'Total items: {itemnum}')
    logger.info(f'Max Time: {timenum}')
    logger.info(f"Creating Model: {args.model_args.get('name', args.model_args.backbone)} "
                f"(backbone={args.model_args.backbone})")

    sampler = WarpSampler(user_train, usernum, itemnum, args, batch_size=args.training_args.batch_size, maxlen=args.model_args.maxlen, n_workers=3)
    
    # TODO pull this out into a seperate function and generalize for model
    if args.model_args.backbone == 'sas':
        logger.info("Eval Neg Samples: {}".format(args.training_args.eval_neg_sample))
        model = SASRec(usernum, itemnum, args).to(args.training_args.device)  # no ReLU activation in original SASRec implementation?
    elif args.model_args.backbone == 'longer':
        logger.info("Eval Neg Samples: {}".format(args.training_args.eval_neg_sample))
        model = LONGER(usernum, itemnum, args).to(args.training_args.device)
    elif args.model_args.backbone == 'gru4rec':
        logger.info("Eval Neg Samples: {}".format(args.training_args.eval_neg_sample))
        model = GRU4RecWrapper(usernum, itemnum, args).to(args.training_args.device)
    elif args.model_args.backbone == 'hstu':
        logger.info("Eval Neg Samples: {}".format(args.training_args.eval_neg_sample))
        model = HSTU(usernum, itemnum, args).to(args.training_args.device)
    elif args.model_args.backbone == 'dp_rec':
        if args.model_args.train_patcher == True:
            model = BehavioralBoundaryModel(
                vocab_size=itemnum + 1,
                dim=args.model_args.hidden_units,
                n_layers=args.model_args.num_blocks,
                n_heads=args.model_args.num_heads,
                max_seqlen=args.model_args.maxlen,
                attn_window=args.model_args.sliding_window,
                loss=args.training_args.loss,
                use_time_rope=getattr(args.model_args, 'use_time_rope', False),
            ).to(args.training_args.device)
            json.dump({'entropy_model': {
                'dim': args.model_args.hidden_units,
                'n_layers': args.model_args.num_blocks,
                'n_heads': args.model_args.num_heads,
                'max_seqlen': args.model_args.maxlen,
                'vocab_size': itemnum + 1,
                'attn_window': args.model_args.sliding_window,
                'loss': args.training_args.loss,
                'use_time_rope': getattr(args.model_args, 'use_time_rope', False),
            }}, open(os.path.join(experiment_path, 'params.json'), 'w'))
            args.model_args.backbone = 'entropy_dp_rec'
        else:
            logger.info("Eval Neg Samples: {}".format(args.training_args.eval_neg_sample))
            args.model_args.vocab_size = itemnum + 1
            if 'expected_patches' in args.training_args:
                logger.info("Expected patches: {}".format(args.training_args.expected_patches))
            model = DPREC(usernum, itemnum, args)
            if getattr(model.detector, "model", None) is not None:
                logger.info(f'BehavioralBoundaryModel Parameters count train/eval: {count_parameters(model.detector.model)}')

    if args.model_args.backbone != 'gru4rec':
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue  # skip frozen patcher weights (e.g. DPREC.detector)
            try:
                torch.nn.init.xavier_normal_(param.data)
            except:
                pass
    
    logger.info(f'Parameters count train/eval: {count_parameters(model)}')

    # GPU Diagnostics
    logger.info(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
        logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    logger.info(f"Model device (first parameter): {next(model.parameters()).device}")

    print(model)
    model.train()

    epoch_start_idx = 1

    bce_criterion = torch.nn.BCEWithLogitsLoss()  # torch.nn.BCELoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.training_args.lr)
    ce_criterion = torch.nn.CrossEntropyLoss()

    T = 0.0
    inference_time = 0.0
    best_ndcg10 = 0

    # Compute inference FLOPs once (config + dataset only; independent of training state)
    try:
        flops_stats = compute_flops(args, dataset)
        if flops_stats:
            logger.info(f"FLOPs: mean={flops_stats['mean_flops']:,.0f}  "
                        f"p99={flops_stats['p99_flops']:,.0f}  "
                        f"total={flops_stats['total_flops']:,.0f}")
    except Exception as e:
        logger.info(f"FLOPs computation skipped: {e}")
        flops_stats = None

    # Calculate validation interval
    if args.training_args.val_interval < 1.0:
        # Percentage of total epochs
        val_epoch_interval = max(1, int(args.training_args.num_epochs * args.training_args.val_interval))
    else:
        # Fixed epoch interval
        val_epoch_interval = int(args.training_args.val_interval)

    logger.info(f"Validation will run every {val_epoch_interval} epoch(s)")

    # Training Loop
    for epoch in range(epoch_start_idx, args.training_args.num_epochs + 1):
        loss_ = []
        for step in tqdm((range(num_batch))):
            u, seq, time_seq, time_matrix, pos, neg = sampler.next_batch() # tuples to ndarray
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
            time_seq, time_matrix = np.array(time_seq), np.array(time_matrix)
            t0 = time.time()
            if args.model_args.backbone == 'entropy_dp_rec':
                ts = time_seq if getattr(args.model_args, 'use_time_rope', False) else None
                pos_logits, neg_logits = model(
                    item_ids=torch.tensor(seq, device=args.training_args.device).long(),
                    pos_seqs=torch.tensor(pos, device=args.training_args.device).long(),
                    neg_seqs=torch.tensor(neg, device=args.training_args.device).long(),
                    time_seq=torch.tensor(ts, device=args.training_args.device).float() if ts is not None else None,
                )
            elif getattr(args.model_args, 'use_time_rope', False) and args.model_args.backbone == 'dp_rec':
                pos_logits, neg_logits = model(u, seq, pos, neg, time_seq=time_seq)
            elif args.model_args.backbone in ['hstu', 'longer']:
                pos_logits, neg_logits = model(u, seq, time_seq, pos, neg)
            else:
                pos_logits, neg_logits = model(u, seq, pos, neg)
            t1 = time.time()
            T += (t1 - t0)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.training_args.device), torch.zeros(neg_logits.shape,
                                                                                                   device=args.training_args.device)
            adam_optimizer.zero_grad()
            indices = np.where(pos != 0)
            if args.training_args.loss == 'bpr':
                loss = bce_criterion(pos_logits[indices], pos_labels[indices])
                loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            elif args.training_args.loss == 'bce':
                y = torch.tensor(pos[indices]).long().to(args.training_args.device)
                yhat = pos_logits[indices]
                loss = ce_criterion(yhat, y)
            else:
                raise Exception('Invalid Loss')
            
            if args.model_args.backbone in ('sas','longer','gru4rec', 'hstu'):
                for param in model.item_emb.parameters(): loss += args.model_args.l2_emb * torch.norm(param)
            t0 = time.time()
            loss.backward()
            t1 = time.time()
            T += (t1 - t0)
            adam_optimizer.step()
            loss_.append(loss.item())

        logger.info(f"Loss in epoch {epoch} iteration {step}: {np.mean(loss_):.6f}")

        # Evaluate on Validation Set
        # Run validation at specified intervals and always at the final epoch
        if epoch % val_epoch_interval == 0 or epoch == args.training_args.num_epochs:

            model.eval()

            logger.info(f'Evaluating validation set at epoch {epoch}/{args.training_args.num_epochs}...')
            val_metrics, T = evaluate(model, dataset, args)
            inference_time += T

            logger.info('Evaluating test set...')
            test_metrics = evaluate_valid(model, dataset, args)

            model.train() # switch back to training mode

            # Always log current epoch metrics
            current_val = {f'Valid_{k}': f'{v:.4f}' for k, v in val_metrics.items()}
            current_test = {f'Test_{k}': f'{v:.4f}' for k, v in test_metrics.items()}
            current_result = {**current_val, **current_test, 'Epoch': str(epoch), 'loss': f'{np.mean(loss_):.6f}'}
            logger.info(f"Epoch {epoch} metrics: {current_result}")

            m_size = get_model_memory_size(model)
            d_size = 0

            # Save checkpoint and results when NDCG@10 improves
            if val_metrics['NDCG@10'] > best_ndcg10:
                best_ndcg10 = val_metrics['NDCG@10']
                torch.save(model.state_dict(), os.path.join(experiment_path, 'state_dict.pth'))
                d_size = get_model_disk_size(os.path.join(experiment_path, 'state_dict.pth'))
                logger.info(f"New best NDCG@10: {best_ndcg10:.4f} at epoch {epoch}")

                write_metrics(
                    logger,
                    loss=np.mean(loss_),
                    epoch=epoch,
                    T=T,
                    val_metrics=val_metrics,
                    test_metrics=test_metrics,
                    model=model,
                    fname=os.path.join(experiment_path, 'results.csv'),
                    m_size=m_size,
                    d_size=d_size,
                    flops=flops_stats
                )

    logger.info(f"Training completed - Total time: {T:.2f}s, Inference time: {inference_time:.2f}s, Parameters: {count_parameters(model)}")

    sampler.close()
    print("Done")


def get_model_memory_size(model):
    # Calculate Parameter Size
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        
    # Calculate Buffer Size (e.g., BatchNorm running mean/var)
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    total_size_bytes = param_size + buffer_size
    total_size_mb = total_size_bytes / (1024 * 1024)
    
    print(f"Model Parameters: {total_size_mb:.2f} MB")
    return total_size_mb


def get_model_disk_size(file_path):
    # Get size in bytes
    size_bytes = os.path.getsize(file_path)
    
    # Convert to MB
    size_mb = size_bytes / (1024 * 1024)
    print(f"Model Disk Size: {size_mb:.2f} MB")
    return size_mb


def inference(args, dataset, experiment_path, logger):
    """
    Run inference on test set using a pre-trained model.
    """
    [user_train, user_valid, user_test, usernum, itemnum, timenum] = dataset

    logger.info(f"Total test sequences: {len(user_test)}")
    cc = 0.0
    for u in user_test:
        cc += len(user_train[u]) + len(user_valid[u])
    logger.info(f'Average sequence length: {cc / len(user_test):.2f}')
    logger.info(f'Total items: {itemnum}')

    logger.info("Creating Model...")
    _train_dir_map = {'sasrec': 'sas', 'dp_rec': 'dp_rec', 'gru4rec': 'gru4rec',
                      'hstu': 'hstu', 'longer': 'longer'}
    _prefix_map = {'sas': 'sas', 'longer': 'longer', 'gru4rec': 'gru4rec',
                   'hstu': 'hstu', 'dp_rec': 'dp_rec', 'entropy_dp_rec': 'dp_rec'}
    def _backbone_from_dir(model_dir):
        name = os.path.basename(model_dir.rstrip('/'))
        for prefix, bb in _prefix_map.items():
            if name.startswith(prefix):
                return bb
        return None
    backbone = (args.model_args.get('backbone')
                or args.model_args.get('name')
                or _train_dir_map.get(args.model_args.get('train_dir'))
                or _backbone_from_dir(args.model_dir))
    if backbone is None:
        raise ValueError(
            f"Could not determine model backbone. model_args={dict(args.model_args)}\n"
            "Add backbone: <name> to your model config."
        )
    if backbone == 'sas':
        model = SASRec(usernum, itemnum, args).to(args.training_args.device)
    elif backbone == 'longer':
        model = LONGER(usernum, itemnum, args).to(args.training_args.device)
    elif backbone == 'gru4rec':
        model = GRU4RecWrapper(usernum, itemnum, args).to(args.training_args.device)
    elif backbone == 'hstu':
        model = HSTU(usernum, itemnum, args).to(args.training_args.device)
    elif backbone == 'dp_rec':
        args.model_args.vocab_size = itemnum + 1
        model = DPREC(usernum, itemnum, args)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    model_path = os.path.join(args.model_dir, 'state_dict.pth')
    logger.info(f"Loading model from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model.load_state_dict(torch.load(model_path, weights_only=False), strict=True)
    model.eval()

    logger.info(f'Parameters count train/eval: {count_parameters(model)}')

    start_time = time.time()

    logger.info('Evaluating validation set...')
    val_metrics, val_time = evaluate(model, dataset, args)

    logger.info('Evaluating test set...')
    test_metrics = evaluate_valid(model, dataset, args)

    total_inference_time = time.time() - start_time

    m_size = get_model_memory_size(model)
    d_size = get_model_disk_size(model_path)

    try:
        flops_stats = compute_flops(args, dataset)
    except Exception as e:
        logger.info(f"FLOPs computation skipped: {e}")
        flops_stats = None

    write_metrics(
        logger=logger,
        loss=0.0,
        epoch=0,
        T=total_inference_time,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        model=model,
        fname=os.path.join(experiment_path, 'inference_results.csv'),
        m_size=m_size,
        d_size=d_size,
        flops=flops_stats
    )

    logger.info(f"Inference completed - Total time: {total_inference_time:.2f}s")


def create_experiment_dir(args):

    # Create Paths:

    if args.task == 'train': 
        experiment_name, experiment_path = create_experiment_folder(args)
        print(f"Model/Inference save location: {experiment_path}")
        # Save configuration
        with open(os.path.join(experiment_path, 'config.yaml'), 'w') as f:
            f.write(OmegaConf.to_yaml(args))
    elif args.task == 'inference_only':
        # Use results_dir if provided, otherwise use model_dir
        if args.results_dir is not None:
            experiment_path = args.results_dir
        else:
            experiment_path = args.model_dir
        
        experiment_name = os.path.basename(experiment_path.rstrip('/'))
        
        # Create directory if it doesn't exist (for local paths)
        if not experiment_path.startswith('s3://'):
            os.makedirs(experiment_path, exist_ok=True)
        
        print(f"Inference results save location: {experiment_path}")
    else:
        raise ValueError(f"Invalid task: {args.task}")

    # Create or Load Results File
    if args.task == 'inference_only':
        logging_path = os.path.join(experiment_path, 'inference_logs.txt')
    else:
        logging_path = os.path.join(experiment_path, 'logs.txt')

    print(f'Logs located at: {logging_path}')

    return experiment_path, experiment_name, logging_path


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg: DictConfig) -> None:
    print("=" * 80)
    print("Configuration:")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    OmegaConf.set_struct(cfg, False)

    # For inference, load the checkpoint's config.yaml and overlay CLI inference keys
    if cfg.task == 'inference_only':
        if cfg.model_dir is None:
            raise ValueError("model_dir must be specified for inference task.")
        ckpt_config_path = os.path.join(cfg.model_dir, 'config.yaml')
        if not os.path.exists(ckpt_config_path):
            raise FileNotFoundError(f"config.yaml not found in model_dir: {cfg.model_dir}")
        ckpt_cfg = OmegaConf.load(ckpt_config_path)
        OmegaConf.set_struct(ckpt_cfg, False)
        # Overlay inference-specific CLI keys onto the checkpoint config
        ckpt_cfg.task = 'inference_only'
        ckpt_cfg.model_dir = cfg.model_dir
        ckpt_cfg.local_data_dir = cfg.local_data_dir
        ckpt_cfg.dataset = cfg.dataset
        if cfg.get('results_dir') is not None:
            ckpt_cfg.results_dir = cfg.results_dir
        cfg = ckpt_cfg

    # Back-fill defaults that may be absent in old checkpoint configs
    cfg.training_args.setdefault('device', cfg.get('device', 'cuda'))
    cfg.training_args.setdefault('loss', 'bpr')
    cfg.training_args.setdefault('eval_neg_sample', 1000)
    cfg.training_args.setdefault('batch_size', 128)
    cfg.model_args.setdefault('backbone', cfg.model_args.get('name'))
    cfg.model_args.setdefault('l2_emb', 0.0)
    cfg.model_args.setdefault('dropout_rate', 0.2)
    cfg.setdefault('results_dir', None)
    
    # Validate required args for training
    if cfg.task == 'train':
        required_args = ["experiment_folder", "model_suffix"]
        missing_args = [arg for arg in required_args if OmegaConf.is_missing(cfg, arg)]
        if len(missing_args) > 0:
            raise ValueError(f"Missing required configuration arguments: {', '.join(missing_args)}")
        # A pre-trained patcher is only needed for entropy patching. Fixed/random
        # strategies build boundaries with no model, so they need no checkpoint.
        _strategy = cfg.model_args.get('patching_strategy', 'entropy')
        if (cfg.model_args.backbone == 'dp_rec' and _strategy == 'entropy'
                and not cfg.model_args.get('train_patcher', False)
                and cfg.get('patcher_model_path') is None):
            raise ValueError("patcher_model_path must be specified for dp_rec training with entropy patching.")
    
    # Convert OmegaConf to namespace for compatibility with existing code
    args = cfg

    # Results and model logging:
    experiment_path, experiment_name, logging_path = create_experiment_dir(args)

    logger = setup_logger(logging_path)

    # Load dataset
    data_file = os.path.join(args.local_data_dir, args.dataset + ".txt") # TODO fix this later
    dataset = data_partition(data_file)

    if args.task=='train':
        train(args, dataset, experiment_path, logger)
    elif args.task=='inference_only':
        inference(args, dataset, experiment_path, logger)
    else:
        raise ValueError("Invalid task specified. Use 'train' or 'inference_only'.")

if __name__ == '__main__':
    main()