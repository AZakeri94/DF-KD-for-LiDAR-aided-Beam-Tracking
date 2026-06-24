"""
train_teachers_only.py
======================
Train the teacher model once for each of the 20 fixed seeds and persist:
    saved_models/teachers/seed_<SEED>/teacher.pth
    saved_models/teachers/seed_<SEED>/metadata.pt
    saved_models/teachers/seed_<SEED>/train_seqs.csv
    saved_models/teachers/seed_<SEED>/val_seqs.csv
    saved_models/teachers/seed_<SEED>/test_seqs.csv

After this script completes you can point main_teacher_KD.py at the
directory by setting  PRETRAINED_TEACHERS_DIR = "./saved_models/teachers"
and it will load the pre-trained teachers instead of re-training them.
"""

import os, time
import torch
import numpy as np

# Re-use helpers from the main orchestrator
import main_teacher_KD as _mk

device      = _mk.device
set_seed    = _mk.set_seed
generate_splits  = _mk.generate_splits
make_loaders     = _mk.make_loaders
train_teacher    = _mk.train_teacher
collect_metadata = _mk.collect_metadata

# ------------------------------------------------------------------ #
SEEDS = [2832, 1198, 8534, 6494, 8691, 4046, 3394, 1785, 9884, 2403,
         4984,  464,   10, 3531, 8895, 6601, 8147, 5751, 1503,  835]

SEQ_LEN       = 8
NUM_EPOCHS_T  = 100
PATIENCE_T    = 5
PORTION       = 1.0

TEACHERS_ROOT = "./saved_models/teachers"
# ------------------------------------------------------------------ #

os.makedirs(TEACHERS_ROOT, exist_ok=True)

t_start = time.time()
for i, seed in enumerate(SEEDS):
    seed_dir = os.path.join(TEACHERS_ROOT, f"seed_{seed}")

    # Skip if already done
    if (os.path.isfile(os.path.join(seed_dir, "teacher.pth")) and
            os.path.isfile(os.path.join(seed_dir, "metadata.pt"))):
        print(f"\n[{i+1}/{len(SEEDS)}] seed={seed}  -> already exists, skipping.")
        continue

    os.makedirs(seed_dir, exist_ok=True)

    print(f"\n{'='*55}\n  TEACHER  {i+1}/{len(SEEDS)}  (seed={seed})\n{'='*55}")

    # 1. Generate data splits (CSV files go into seed_dir)
    generate_splits(seed, csv_dir=seed_dir, seq_length=SEQ_LEN)

    # 2. Build dataloaders
    set_seed(seed)
    train_loader, val_loader, _ = make_loaders(PORTION, SEQ_LEN, csv_dir=seed_dir)

    # 3. Train teacher  (train_teacher writes to _mk.save_dir which we override)
    _mk.save_dir = seed_dir          # redirect checkpoint writes
    teacher, t_top1, t_top5 = train_teacher(
        train_loader, val_loader, num_classes=64,
        num_epochs=NUM_EPOCHS_T, patience=PATIENCE_T
    )
    print(f"  Teacher Top-1 (val): {np.round(t_top1, 4)}")
    print(f"  Teacher Top-5 (val): {np.round(t_top5, 4)}")

    # 4. Collect metadata for generator training
    metadata = collect_metadata(teacher, train_loader)   # saves metadata.pt
    print(f"  Metadata saved to {seed_dir}/metadata.pt")

elapsed = time.time() - t_start
print(f"\n{'='*55}")
print(f"  All teachers trained.  Total time: {elapsed/60:.1f} min")
print(f"  Saved under: {TEACHERS_ROOT}")
print(f"{'='*55}")
