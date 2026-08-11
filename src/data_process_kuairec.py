from tqdm import tqdm
import collections
import os
import numpy as np

# Ensure data directory exists
if not os.path.exists('data'):
    os.makedirs('data')

def load_ratings(file):
    """
    Loads ratings and converts raw millisecond timestamps to epoch seconds.
    """
    inters = []
    # # --- Global time baseline scan (commented out: now using epoch seconds) ---
    # # Find the minimum timestamp in the file to use as a global baseline (timeSlice logic)
    # print("Scanning for global time baseline (timeSlice)...")
    # global_min_time = None
    #
    # # First pass to find global min
    # with open(file, 'r') as fp:
    #     for count, line in enumerate(fp):
    #         if count == 0: continue
    #         parts = line.strip().split(',')
    #         if len(parts) < 6: continue
    #
    #         time_val = int(parts[4]) # raw time_ms
    #         if global_min_time is None or time_val < global_min_time:
    #             global_min_time = time_val

    # Single pass: convert ms to epoch seconds directly
    with open(file, 'r') as fp:
        for count, line in enumerate(tqdm(fp, desc='Load ratings')):
            if count == 0: continue
            parts = line.strip().split(',')
            if len(parts) < 6: continue

            user, item, _, _, time_ms, is_click = parts[:6]
            if is_click == '1':
                # Convert ms to epoch seconds
                epoch_sec = int(round(float(int(time_ms)) / 1000.0))
                # # --- Global offset (commented out: now using epoch seconds) ---
                # reduced_time_sec = int(round(float(int(time_ms) - global_min_time) / 1000.0))
                inters.append((user, item, epoch_sec))
    return inters


def load_all_ratings(path):
    """Load interactions from a single CSV, or merge all standard-log parts.

    KuaiRand-27K ships the standard log split across parts, e.g.
    log_standard_4_08_to_4_21_27k_part1.csv / _part2.csv and the 4_22_to_5_08
    equivalents. If `path` is a directory, every `log_standard_*.csv` in it is
    loaded and concatenated (replacing the manual pandas merge into
    full_data.csv). The random-exposure log (log_random_*) is intentionally
    excluded. If `path` is a file, it is loaded directly.
    """
    import glob
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, 'log_standard_*.csv')))
        if not files:
            raise FileNotFoundError(
                f"No 'log_standard_*.csv' files found in directory: {path}. "
                "Pass a merged CSV file instead, or point at the KuaiRand-27K "
                "data directory that contains the standard-log parts."
            )
        print(f"Merging {len(files)} standard-log part file(s):")
        for f in files:
            print(f"  - {os.path.basename(f)}")
        inters = []
        for f in files:
            inters.extend(load_ratings(f))
        return inters
    return load_ratings(path)

def process_data_with_time(inters):
    """
    Groups interactions by user, remaps IDs, and sorts by timestamp.
    Timestamps are kept as epoch seconds.
    """
    user_set = set()
    item_set = set()
    raw_user_inters = collections.defaultdict(list)

    for u, i, t in inters:
        user_set.add(u)
        item_set.add(i)
        raw_user_inters[u].append([i, t])

    # Remap IDs to 1-indexed contiguous range (consistent with cleanAndsort)
    user_map = {user: u + 1 for u, user in enumerate(user_set)}
    item_map = {item: i + 1 for i, item in enumerate(item_set)}

    processed_user_inters = {}

    for raw_u, items in tqdm(raw_user_inters.items(), desc='Processing users'):
        # Sort by timestamp
        items.sort(key=lambda x: x[1])

        u_mapped = user_map[raw_u]

        # Keep original Unix epoch timestamps (matching RecTools)
        # This preserves global temporal relationships for relative time attention
        processed_inters = [[item_map[x[0]], x[1]] for x in items]

        # # --- Per-user time normalization (DO NOT USE: breaks global temporal relationships) ---
        # # This was normalizing timestamps to seconds since user's first interaction
        # # but HSTU's relative time attention needs global timestamps to work correctly
        # timestamps = [x[1] for x in items]
        # time_min = min(timestamps)
        # processed_inters = [[item_map[x[0]], x[1] - time_min] for x in items]

        # # --- Per-user time normalization (commented out: now using epoch seconds) ---
        # item_ids = [item_map[x[0]] for x in items]
        # timestamps = [x[1] for x in items]
        #
        # # Calculate time_scale (min non-zero delta)
        # time_diffs = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)
        #               if timestamps[i+1] - timestamps[i] != 0]
        #
        # time_scale = min(time_diffs) if len(time_diffs) > 0 else 1
        # time_min = min(timestamps)
        #
        # normalized_inters = []
        # for i_mapped, t_offset in zip(item_ids, timestamps):
        #     # Per-user normalization logic: int(round((time - min) / scale) + 1)
        #     t_norm = int(round((t_offset - time_min) / time_scale) + 1)
        #
        #     # Int32 Safety Check
        #     if t_norm > 2147483647:
        #         t_norm = 2147483647
        #
        #     normalized_inters.append([i_mapped, t_norm])

        processed_user_inters[u_mapped] = processed_inters

    return processed_user_inters, len(user_set), len(item_set)

def write_results(user2inters, base_path):
    lengths = [None]
    for length in lengths:
        label = length - 2 if length is not None else 'full'
        output_file = f'{base_path}_{label}_timed.txt'
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)

        sequence_lengths = []
        item_repeat_rates = []

        with open(output_file, 'w') as f:
            for user in tqdm(user2inters, desc=f'Writing length={label}'):
                user_items = user2inters[user] if length is None else user2inters[user][-length:]

                # Track sequence length
                sequence_lengths.append(len(user_items))

                # Calculate repeat rate for this user
                total_items = len(user_items)
                unique_items = len(set([item for item, time in user_items]))
                item_repeat_rate = (total_items - unique_items) / total_items if total_items > 0 else 0
                item_repeat_rates.append(item_repeat_rate)

                # Write interactions
                for item, time in user_items:
                    f.write(f"{user} {item} {time}\n")

        # Print statistics for this length variant
        print(f"\n=== Statistics for {output_file} ===")
        print(f"Number of users: {len(sequence_lengths)}")
        print(f"Average sequence length: {sum(sequence_lengths) / len(sequence_lengths):.2f}")
        print(f"Min sequence length: {min(sequence_lengths)}")
        print(f"Max sequence length: {max(sequence_lengths)}")
        avg_item_repeat = sum(item_repeat_rates) / len(item_repeat_rates) * 100
        print(f"Average item repeat rate: {avg_item_repeat:.2f}%")

if __name__ == "__main__":
    import sys
    # Input path from the command line (resolved against the current directory,
    # so this works whether run from DP-Rec or DP-Rec-Exp). May be a single
    # merged CSV or the KuaiRand-27K data directory (log_standard_* parts are
    # auto-merged). Falls back to the default cluster data dir with no argument.
    raw_file = sys.argv[1] if len(sys.argv) > 1 else "/home/jovyan/datavol/datasets/KuaiRand-27K/data"
    
    # Step 1: Global Time Baseline (Anchors dataset to 0)
    inters = load_all_ratings(raw_file)
    
    # Step 2: ID remapping and sorting (timestamps kept as epoch seconds)
    user2inters_processed, usernum, itemnum = process_data_with_time(inters)
    # # --- Old call with per-user normalization ---
    # user2inters_processed, usernum, itemnum, timenum = process_data_with_time(inters)

    print(f"\nProcessing Complete:")
    print(f"Users: {usernum}, Items: {itemnum}")
    
    # Step 3: Write out results (data/ under the current directory, matching the
    # dataset=<name> → data/<name>.txt convention; created if missing)
    write_results(user2inters_processed, 'data/KuaiRand-27K')