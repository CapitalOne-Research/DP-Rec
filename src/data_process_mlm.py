#!/usr/bin/env python3
"""
Process MovieLens datasets (ML-1M or ML-10M) into DP-Rec format.

Input:  ratings.dat from the original MovieLens archive
Output: <dataset>.txt in the data/ directory

Usage:
    python data_process_mlm.py ml-1m/ratings.dat
    python data_process_mlm.py ml-10M100K/ratings.dat --dataset-name ml-10m
    python data_process_mlm.py ml-10M100K/ratings.dat --dataset-name ml-10m --min-seq-len 500
    python data_process_mlm.py ml-1m/ratings.dat --inspect

Options:
    --dataset-name S  Output name S.txt (default: derived from the ratings folder).
                      ml-10m.zip extracts to ml-10M100K/, so pass --dataset-name ml-10m.
    --min-seq-len N   Keep only users with at least N interactions (default: 5).
    --inspect         Inspect the processed output file instead of processing.
"""

import sys
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path


def cleanAndsort(User):
    user_set = set(User.keys())
    item_set = set()
    for items in User.values():
        for item in items:
            item_set.add(item[0])

    user_map = {user: u + 1 for u, user in enumerate(user_set)}
    item_map = {item: i + 1 for i, item in enumerate(item_set)}

    User_res = {}
    for user, items in User.items():
        sorted_items = sorted(items, key=lambda x: x[1])
        User_res[user_map[user]] = [[item_map[x[0]], int(x[1])] for x in sorted_items]

    return User_res, len(user_set), len(item_set)


def read_ratings(ratings_file):
    """Read ratings.dat, returning (user, item, timestamp) triples."""
    records = []
    with open(ratings_file, 'r') as f:
        for line in f:
            parts = line.rstrip().split('::')
            if len(parts) >= 4:
                # ML-1M / ML-10M format: UserID::MovieID::Rating::Timestamp
                user, item, rating, timestamp = parts[:4]
            else:
                # Fallback: tab-separated with or without rating column
                cols = line.rstrip().split('\t')
                if len(cols) == 4:
                    user, item, rating, timestamp = cols
                elif len(cols) == 3:
                    user, item, timestamp = cols
                else:
                    continue
            records.append((int(user), int(item), float(timestamp)))
    return records


def process(ratings_file, output_file, min_seq_len=5):
    ratings_file = Path(ratings_file)
    output_file = Path(output_file)

    if not ratings_file.exists():
        print(f"ERROR: File not found: {ratings_file}")
        sys.exit(1)

    print(f"Reading {ratings_file} ...")
    records = read_ratings(ratings_file)
    print(f"Loaded {len(records):,} ratings")

    # Count interactions per user
    user_count = defaultdict(int)
    for u, i, ts in records:
        user_count[u] += 1

    # Build user history (keep users with >= min_seq_len interactions)
    User = defaultdict(list)
    for u, i, ts in tqdm(records, desc="Filtering"):
        if user_count[u] >= min_seq_len:
            User[u].append([i, ts])

    print(f"Users after filtering (>={min_seq_len} interactions): {len(User):,}")

    User, usernum, itemnum = cleanAndsort(User)

    print(f"Remapped users: {usernum:,}, items: {itemnum:,}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        for user, items in tqdm(User.items(), desc="Writing"):
            for item, time in items:
                f.write(f"{user} {item} {time}\n")

    print(f"Written to: {output_file}")


def inspect(output_file):
    output_file = Path(output_file)
    if not output_file.exists():
        print(f"File not found: {output_file}")
        sys.exit(1)

    users, items, timestamps = set(), set(), []
    with open(output_file, 'r') as f:
        lines = f.readlines()

    print(f"File: {output_file}  ({len(lines):,} lines)")
    print("\nFirst 10 lines:")
    for line in lines[:10]:
        parts = line.strip().split()
        if len(parts) == 3:
            u, i, ts = parts
            print(f"  user={u:>5}  item={i:>5}  ts={ts}")

    for line in lines:
        parts = line.strip().split()
        if len(parts) == 3:
            u, i, ts = parts
            users.add(int(u))
            items.add(int(i))
            timestamps.append(int(float(ts)))

    print(f"\nUnique users : {len(users):,}")
    print(f"Unique items : {len(items):,}")
    print(f"Total records: {len(timestamps):,}")
    if timestamps:
        import pandas as pd
        print(f"Timestamp min: {min(timestamps)}  ({pd.to_datetime(min(timestamps), unit='s')})")
        print(f"Timestamp max: {max(timestamps)}  ({pd.to_datetime(max(timestamps), unit='s')})")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description="Process MovieLens (ML-1M / ML-10M) into DP-Rec format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("ratings_file", help="Path to ratings.dat (e.g. ml-10M100K/ratings.dat)")
    parser.add_argument("--dataset-name", default=None,
                        help="Output dataset name (default: derived from the ratings folder). "
                             "Use e.g. --dataset-name ml-10m since ml-10m.zip extracts to ml-10M100K/")
    parser.add_argument("--min-seq-len", type=int, default=5,
                        help="Keep only users with at least this many interactions")
    parser.add_argument("--inspect", action="store_true",
                        help="Inspect the processed output file instead of processing")
    args = parser.parse_args()

    ratings_path = Path(args.ratings_file)

    # Output name: explicit --dataset-name, else derived from the ratings folder
    # (e.g. ml-1m/ratings.dat -> ml-1m). Note: ml-10m.zip extracts to ml-10M100K/,
    # so pass --dataset-name ml-10m to keep the training convention (dataset=ml-10m).
    dataset_name = args.dataset_name or ratings_path.parent.name
    output_path = ratings_path.parent.parent.parent / "data" / f"{dataset_name}.txt"

    if args.inspect:
        inspect(output_path)
    else:
        process(ratings_path, output_path, min_seq_len=args.min_seq_len)
