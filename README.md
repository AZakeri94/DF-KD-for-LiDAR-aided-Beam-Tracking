# Data-Free Knowledge Distillation for mmWave Beam Selection

A Knowledge Distillation (KD) framework for 60 GHz mmWave beam selection using LiDAR sequences. The framework includes a Data-Free KD (DF-KD) variant in which a generator network synthesizes training samples, removing the need to access the original dataset during student training.

## Problem Statement

Given a sequence of 8 LiDAR frames (216 features each), predict the best beam index among 64 candidates for the next 3 time steps. The goal is to compress a larger teacher model into a lightweight student model suitable for deployment.

## Methods Compared

| Method | Real Data | Knowledge Transfer |
|---|---|---|
| Vanilla Student | Yes | None |
| KD Student | Yes | Soft labels (KL divergence) |
| KD + Features | Yes | Soft labels + feature alignment |
| DF-KD Student | No (synthetic) | Soft labels from generator |

## Model Architectures

- Teacher: GRU-CNN hybrid — 1D CNN for spatial feature extraction + GRU for temporal modeling (hidden=128, embed=64)
- Student: Lightweight GRU (hidden=24, embed=16, ~4–5× smaller than teacher)
- Generator: MLP mapping 500-D Gaussian noise → synthetic LiDAR sequences (8×216)

## Dataset

"DeepSense 6G — Scenario 8"
LiDAR point cloud data paired with 60 GHz beam power measurements.
Download from: [https://www.deepsense6g.net/](https://www.deepsense6g.net/)

After downloading, update the dataset path in `main_teacher_KD.py`:
```python
data_path = "/your/path/to/DeepSense/Scenario8/"
```

## Requirements

```bash
pip install torch numpy pandas scipy scikit-image
```

## Usage

Step 1 — Generate train/val/test splits:
```bash
python utils/gen_data_seq.py
```

Step 2 — Run full pipeline (teacher + all student variants):
```bash
python main_teacher_KD.py
```

Step 3 — Data-Free KD only:
```bash
python main_DF_KD.py
```

Pre-train teachers across multiple seeds:
```bash
python train_teachers_only.py
```

## Project Structure

```
├── main_teacher_KD.py             # Main pipeline: teacher + all student variants
├── main_DF_KD.py                  # Data-Free KD: generator training + DF student
├── model.py                       # Model architectures (teacher, student, generator)
├── train_teachers_only.py         # Pre-train teacher models for multiple seeds
├── extract_results.py             # Parse final_results.log and print summary table
├── run_script_tvt.sh              # SLURM batch submission script (A100 GPU)
├── loss_functions/
│   ├── data_generator_lossess.py  # Generator loss: match teacher feature statistics
│   └── training_student_lossess.py# KD loss: temperature-scaled KL divergence
├── utils/
│   ├── data_feed.py               # PyTorch Dataset loader (.mat files)
│   ├── evl_topK.py                # Top-K accuracy evaluation (4 horizons)
│   └── gen_data_seq.py            # Train/val/test CSV split generator
└── saved_versions/                # Timestamped experiment outputs (models + logs)
```

## Evaluation

Results are reported as **Top-1** and **Top-5** accuracy across 4 prediction horizons.
Logs are saved to `final_results.log` inside each timestamped run folder under `saved_versions/`.

## Citation

If you use this code in your research, please cite:

A. Zakeri, N. T. Nguyen, A. Alkhateeb, and M. Juntti, “Data-free
knowledge distillation for LiDAR-aided beam tracking in MmWave
systems,” IEEE Trans. Veh. Technol., Jun. 2026.