#Ref: https://github.com/nancheng58/RecMamba/tree/main

import sys
import copy
import torch
import random
import numpy as np
import time
import logging
import os
import pickle
from collections import defaultdict
from multiprocessing import Process, Queue
from tqdm import tqdm


def setup_logger(log_file_path):
    """Setup logger that writes to both stdout and file"""
    logger = logging.getLogger('DP-REC')
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatters
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(log_file_path, mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

def generate_experiment_id(cfg, ensure_unique=True):
    """Generate a short, unique experiment ID based on model and training configs only

    Args:
        cfg: Configuration object
        ensure_unique: If True, adds timestamp to ensure uniqueness across runs
    """
    # Extract only the relevant config sections
    relevant_config = {
        'model_args': cfg.model_args if hasattr(cfg, 'model_args') else {},
        'training_args': cfg.training_args if hasattr(cfg, 'training_args') else {}
    }

    # Convert to YAML string for hashing
    from omegaconf import OmegaConf
    import hashlib
    import time
    import random

    config_str = OmegaConf.to_yaml(OmegaConf.create(relevant_config))

    if ensure_unique:
        # Add timestamp and random component to ensure uniqueness even for identical configs
        config_str += f"\n_timestamp: {time.time()}\n_random: {random.random()}"

    hash_obj = hashlib.md5(config_str.encode())
    return hash_obj.hexdigest()[:8]  # 8-char unique ID

def get_gpu_type():
    """Get GPU type string for experiment naming"""
    import torch
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
            # Simplify common GPU names
            if 'A100' in gpu_name:
                return 'a100'
            elif 'V100' in gpu_name:
                return 'v100'
            elif 'H100' in gpu_name:
                return 'h100'
            elif 'RTX 4090' in gpu_name or '4090' in gpu_name:
                return '4090'
            elif 'RTX 3090' in gpu_name or '3090' in gpu_name:
                return '3090'
            elif 'T4' in gpu_name:
                return 't4'
            else:
                # Extract first alphanumeric part as fallback
                import re
                match = re.search(r'[A-Za-z0-9]+', gpu_name)
                return match.group(0).lower() if match else 'gpu'
        except:
            return 'gpu'
    else:
        return 'cpu'


def create_experiment_folder(args):
    """Create organized experiment folder structure with model parent folder.

    Expected structure: base/dataset/model/artifact
    Examples:
        - paper/ml-1m/dp_rec/dp_rec_d64x3_32x1
        - paper/kuairec/sasrec/sasrec_baseline
    """
    folder_parts = args.experiment_folder.strip('/').split('/')
    model_name = args.model_args.backbone
    gpu_type = get_gpu_type()

    # Check if model folder is already in the path
    # Structure should be: base/dataset/model/artifact (4 parts)
    if len(folder_parts) >= 4:
        experiment_path = args.experiment_folder
        experiment_name = folder_parts[-1]
    elif len(folder_parts) == 3:
        # Has base/dataset/artifact but missing model folder
        # Check if third part is the model name
        if folder_parts[-1] == model_name or folder_parts[-1].startswith(model_name + '_'):
            # Third part looks like an artifact, insert model folder
            base_path = os.path.join(*folder_parts[:-1])  # paper/ml-1m
            artifact_name = folder_parts[-1]
            experiment_path = os.path.join(base_path, model_name, artifact_name)
            experiment_name = artifact_name
        else:
            # Third part is model folder, missing artifact
            exp_id = generate_experiment_id(args)
            experiment_name = f"{model_name}_{args.model_suffix}_{args.dataset}_{gpu_type}_{exp_id}"
            experiment_path = os.path.join(args.experiment_folder, experiment_name)
    elif len(folder_parts) == 2:
        # Only base/dataset, need to add model/artifact
        exp_id = generate_experiment_id(args)
        experiment_name = f"{model_name}_{args.model_suffix}_{args.dataset}_{gpu_type}_{exp_id}"
        experiment_path = os.path.join(args.experiment_folder, model_name, experiment_name)
    else:
        # Just base or invalid
        exp_id = generate_experiment_id(args)
        experiment_name = f"{model_name}_{args.model_suffix}_{args.dataset}_{gpu_type}_{exp_id}"
        experiment_path = os.path.join(args.experiment_folder, model_name, experiment_name)

    if os.path.exists(experiment_path):
        # Path already exists - add unique suffix to avoid collision
        import random
        import string
        random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
        experiment_name = f"{experiment_name}_{random_suffix}"
        experiment_path = os.path.join(os.path.dirname(experiment_path), experiment_name)
        print(f"Warning: Original path existed. Using unique suffix: {experiment_name}")
        os.makedirs(experiment_path, exist_ok=True)
    else:
        os.makedirs(experiment_path, exist_ok=True)

    return experiment_name, experiment_path

# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

def computeRePos_window(time_seq, time_span, window_size):
    size = time_seq.shape[0]
    time_matrix = np.abs(time_seq[:, np.newaxis] - time_seq[np.newaxis, :])
    indices = np.arange(size)
    pos_dist = np.abs(indices[:, np.newaxis] - indices[np.newaxis, :])
    mask = (time_matrix > time_span)
    if window_size != None:
        mask = mask | (pos_dist > window_size)
    time_matrix[mask] = time_span
    return time_matrix.astype(np.int32)

def Relation(user_train, usernum, maxlen, time_span, window_size):
    data_train = dict()
    for user in tqdm(range(1, usernum+1), desc='Preparing relation matrix'):
        time_seq = np.zeros([maxlen], dtype=np.int32)
        idx = maxlen - 1
        for i in reversed(user_train[user][:-1]):
            time_seq[idx] = i[1]
            idx -= 1
            if idx == -1: break
        data_train[user] = computeRePos_window(time_seq, time_span, window_size)
    return data_train


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, args, result_queue, SEED):
    # Get number of negatives from config (default: 1 for backward compatibility)
    num_negatives = args.training_args.get('num_negatives', 1)

    def sample(user):

        seq = np.zeros([maxlen], dtype=np.int32)
        time_seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)

        # Support multiple negatives
        if num_negatives > 1:
            neg = np.zeros([maxlen, num_negatives], dtype=np.int32)
        else:
            neg = np.zeros([maxlen], dtype=np.int32)

        nxt = user_train[user][-1][0]

        idx = maxlen - 1
        ts = set(map(lambda x: x[0],user_train[user]))
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i[0]
            time_seq[idx] = i[1]
            pos[idx] = nxt
            if nxt != 0:
                if num_negatives > 1:
                    # Sample multiple negatives
                    for neg_idx in range(num_negatives):
                        neg[idx, neg_idx] = random_neq(1, itemnum + 1, ts)
                else:
                    # Single negative (backward compatibility)
                    neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i[0]
            idx -= 1
            if idx == -1: break
        if args.model_args.get('backbone') == 'tisas':
            time_matrix = computeRePos_window(time_seq, args.model_args.get('time_span', 256), args.model_args.get('sliding_window'))
        else:
            time_matrix=None
        return (user, seq, time_seq, time_matrix, pos, neg)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            user = np.random.randint(1, usernum + 1)
            while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)
            one_batch.append(sample(user))

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, args, batch_size=64, maxlen=10,n_workers=1,):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      args,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()
    
# train/val/test data generation
def data_partition(fname):
    # Check for cached partition file
    dataset_name = os.path.basename(fname).replace('.txt', '').replace('.dat', '')
    cache_dir = os.path.join(os.path.dirname(fname), 'partitioned')
    cache_file = os.path.join(cache_dir, f"{dataset_name}_partitioned.pkl")

    # Try to load from cache
    if os.path.exists(cache_file):
        print(f"Loading cached partition from {cache_file}")
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            print(f"Successfully loaded cached partition (usernum={cached_data[3]}, itemnum={cached_data[4]})")
            return cached_data
        except Exception as e:
            print(f"Failed to load cache: {e}. Re-partitioning dataset...")

    # Perform partitioning
    print(f"Partitioning dataset {fname}...")
    usernum = 0
    itemnum = 0
    timenum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    
    if ".dat" in fname:
        # Added encoding='latin-1' because MovieLens .dat files often need it
        with open(fname, 'r', encoding='latin-1') as f:
            for line in f:
                line = line.strip()
                if not line: 
                    continue
                # MovieLens 1M uses '::' as a separator
                parts = line.split('::')
                # ratings.dat has [user, item, rating, timestamp]
                if len(parts) < 2: 
                    continue
                u = int(parts[0])
                i = int(parts[1])
                usernum = max(u, usernum)
                itemnum = max(i, itemnum)
                User[u].append(i)
                
    elif ".txt" in fname:
        # assume user/item index starting from 1
        # fname is already the full path, don't add 'data/' prefix
        f = open(fname, 'r')
        for line in f:
            #u, i = line.rstrip().split(' ')for line in f:
            line = line.strip()
            if not line: continue  # Skip empty lines
            parts = line.split(' ')
            if len(parts) < 2: 
                #print(f"Skipping bad line: {line}")
                continue # Skip malformed lines
            u, i, t = parts
            u = int(u)
            i = int(i)
            
            t = int(t)
            usernum = max(u, usernum)
            itemnum = max(i, itemnum)
            timenum = max(t, timenum)
            User[u].append((i, t))
    
    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])

    # Prepare result
    result = [user_train, user_valid, user_test, usernum, itemnum, timenum]

    # Save to cache
    try:
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
        print(f"Saved partition cache to {cache_file}")
    except Exception as e:
        print(f"Warning: Failed to save partition cache: {e}")

    return result


# TODO: merge evaluate functions for test and val set
# evaluate on test set

def retrieve_pos_neg(user, train, itemnum, valid=None, maxlen=None):
    pos = np.zeros([maxlen], dtype=np.int32)
    neg = np.zeros([maxlen], dtype=np.int32)
    seq = np.zeros([maxlen], dtype=np.int32)
    
    user_seq = train[user] + valid[user] if valid is not None else train[user]

    nxt = user_seq[-1][0]
    idx = maxlen - 1

    ts = set(map(lambda x: x[0],user_seq))
    for i in reversed(user_seq[:-1]):
        seq[idx] = i[0]
        pos[idx] = nxt
        if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
        nxt = i[0]
        idx -= 1
        if idx == -1: break
    
    return pos, neg

    
def evaluate(model, dataset, args):
    [train, valid, test, usernum, itemnum, timenum] = dataset
    sumt = 0
    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0
    NDCG_20, HT_20 = 0.0, 0.0
    NDCG_5, HT_5 = 0.0, 0.0


    users = range(1, usernum + 1)

    # Apply user limit if specified
    max_users = args.training_args.get('eval_max_users_valid', None)
    if max_users is not None:
        users = list(users)[:max_users]

    with torch.no_grad():
        for u in tqdm(users):
            if len(train[u]) < 1 or len(test[u]) < 1: continue

            seq = np.zeros([args.model_args.maxlen], dtype=np.int32)
            time_seq = np.zeros([args.model_args.maxlen], dtype=np.int32)
            idx = args.model_args.maxlen - 1

            seq[idx] = valid[u][0][0]
            time_seq[idx] = valid[u][0][1]
            idx -= 1
            for i in reversed(train[u]):
                seq[idx] = i[0]
                time_seq[idx] = i[1]
                idx -= 1
                if idx == -1: break
            rated = set(map(lambda x: x[0],train[u]))
            rated.add(valid[u][0][0])
            rated.add(test[u][0][0])
            rated.add(0)
            item_idx = [test[u][0][0]]
            num_to_sample = int(itemnum * args.training_args.eval_neg_sample) if args.training_args.eval_neg_sample <= 1 else int(args.training_args.eval_neg_sample)
            for _ in range(num_to_sample):
                t = np.random.randint(1, itemnum + 1)
                while t in rated: t = np.random.randint(1, itemnum + 1)
                item_idx.append(t)

            t0 = time.time()

            if args.model_args.backbone == 'dp_rec' and args.training_args.loss == 'bpr':
                pos, neg = retrieve_pos_neg(u, train=train, valid=valid, itemnum=itemnum, maxlen=args.model_args.maxlen)
                use_time_rope = getattr(args.model_args, 'use_time_rope', False)
                if use_time_rope:
                    predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx, [pos], [neg]]], time_seq=np.array([time_seq]))
                else:
                    predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx, [pos], [neg]]])
            elif args.model_args.backbone == 'tisas':
                time_matrix = computeRePos_window(time_seq, args.model_args.get('time_span', 256), args.model_args.get('sliding_window'))
                predictions = -model.predict(*[np.array(l) for l in [[u], [seq], [time_matrix], item_idx]])
            elif args.model_args.backbone in ['longer', 'hstu']:
                predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]], time_seqs=np.array([time_seq]))
            else:
                use_time_rope_else = getattr(args.model_args, 'use_time_rope', False)
                if use_time_rope_else and 'entropy' in str(args.model_args.backbone):
                    predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]], time_seq=np.array([time_seq]))
                else:
                    predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
            t1 = time.time()
            sumt += (t1 - t0)
            predictions = predictions[0]  # - for 1st argsort DESC

            rank = predictions.argsort().argsort()[0].item()

            valid_user += 1

            if rank < 5:
                NDCG_5 += 1 / np.log2(rank + 2)
                HT_5 += 1


            if rank < 10:
                NDCG += 1 / np.log2(rank + 2)
                HT += 1


            if rank < 20:
                NDCG_20 += 1 / np.log2(rank + 2)
                HT_20 += 1

    results={
        'NDCG@5': NDCG_5 / valid_user,
        'HR@5': HT_5 / valid_user,
        'NDCG@10': NDCG / valid_user,
        'HR@10': HT / valid_user,
        'NDCG@20': NDCG_20 / valid_user,
        'HR@20': HT_20 / valid_user
    }
    return results, sumt



# evaluate on val set
def evaluate_valid(model, dataset, args):
    [train, valid, test, usernum, itemnum, timenum] = dataset

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    NDCG_20, HT_20 = 0.0, 0.0
    NDCG_5, HT_5 = 0.0, 0.0


    users = range(1, usernum + 1)

    # Apply user limit if specified
    max_users = args.training_args.get('eval_max_users_test', None)
    if max_users is not None:
        users = list(users)[:max_users]

    with torch.no_grad():
        for u in tqdm(users):
            if len(train[u]) < 1 or len(valid[u]) < 1: continue

            seq = np.zeros([args.model_args.maxlen], dtype=np.int32)
            time_seq = np.zeros([args.model_args.maxlen], dtype=np.int32)
            idx = args.model_args.maxlen - 1
            for i in reversed(train[u]):
                seq[idx] = i[0]
                time_seq[idx] = i[1]
                idx -= 1
                if idx == -1: break

            rated = set(map(lambda x: x[0], train[u]))
            rated.add(valid[u][0][0])
            rated.add(0)
            item_idx = [valid[u][0][0]]
            num_to_sample = int(itemnum * args.training_args.eval_neg_sample) if args.training_args.eval_neg_sample <= 1 else int(args.training_args.eval_neg_sample)
            for _ in range(num_to_sample):
                t = np.random.randint(1, itemnum + 1)
                while t in rated: t = np.random.randint(1, itemnum + 1)
                item_idx.append(t)

            if args.model_args.backbone == 'dp_rec' and args.training_args.loss == 'bpr':
                pos, neg = retrieve_pos_neg(u, train=train, valid=None, itemnum=itemnum, maxlen=args.model_args.maxlen)
                use_time_rope = getattr(args.model_args, 'use_time_rope', False)
                if use_time_rope:
                    predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx, [pos], [neg]]], time_seq=np.array([time_seq]))
                else:
                    predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx, [pos], [neg]]])
            elif args.model_args.backbone == 'tisas':
                time_matrix = computeRePos_window(time_seq, args.model_args.get('time_span', 256), args.model_args.get('sliding_window'))
                predictions = -model.predict(*[np.array(l) for l in [[u], [seq], [time_matrix], item_idx]])
            elif args.model_args.backbone in ['hstu', 'longer']:
                predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]], time_seqs=np.array([time_seq]))
            else:
                use_time_rope_else = getattr(args.model_args, 'use_time_rope', False)
                if use_time_rope_else and 'entropy' in str(args.model_args.backbone):
                    predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]], time_seq=np.array([time_seq]))
                else:
                    predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])

            predictions = predictions[0]

            rank = predictions.argsort().argsort()[0].item()

            valid_user += 1

            if rank < 5:
                NDCG_5 += 1 / np.log2(rank + 2)
                HT_5 += 1

            if rank < 10:
                NDCG += 1 / np.log2(rank + 2)
                HT += 1


            if rank < 20:
                NDCG_20 += 1 / np.log2(rank + 2)
                HT_20 += 1


    results={
        'NDCG@5': NDCG_5 / valid_user,
        'HR@5': HT_5 / valid_user,
        'NDCG@10': NDCG / valid_user,
        'HR@10': HT / valid_user,
        'NDCG@20': NDCG_20 / valid_user,
        'HR@20': HT_20 / valid_user
    }
    return results
