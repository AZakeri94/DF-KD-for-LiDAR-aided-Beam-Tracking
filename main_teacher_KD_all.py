import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import random
import matplotlib.pyplot as plt

import pandas as pd

from utils.data_feed import DataFeed
from model import GruModel, student_model, gru_cnn_teacher
from utils.evl_topK import evaluate_model
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


import main_DF_KD as _df_kd
generator_model  = _df_kd.generator_model
train_generator          = _df_kd.train_generator
train_student_fixed_data = _df_kd.train_student_fixed_data
input_dim                = _df_kd.input_dim

import time; start = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#import pdb; pdb.set_trace()

save_dir = "./saved_versions"
os.makedirs(save_dir, exist_ok=True)

# =================== Helpers ===================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

#%%
def generate_splits(seed, csv_dir, seq_length=8):
    """Re-generate train/val/test CSV splits with a given seed (sequence-level split)."""

    scenario_num = 8  # <-- change scenario here ONLY

    # Keep data_feed in sync so DataFeed loads .mat files from the correct scenario folder
    import utils.data_feed as _df_mod
    _df_mod.scenario_num = scenario_num
    _df_mod.data_root = f'/projappl/project_2013517/DeepSense/datasets/Scenario{scenario_num}/'

    data_root_csv = f'/projappl/project_2013517/DeepSense/datasets/Scenario{scenario_num}/scenario{scenario_num}.csv'
    in_len  = seq_length
    out_len = 3

    all_data = pd.read_csv(data_root_csv)
    all_seq_idx = all_data['seq_index'].unique()

    all_seq_split = [all_data[all_data['seq_index'] == i][['unit1_pwr_60ghz', 'unit1_lidar_SCR', 'seq_index']]
                     for i in all_seq_idx]

    all_seqs = []
    for seq in all_seq_split:
        start = 0
        while start + in_len <= seq.shape[0]:
            lidar        = seq['unit1_lidar_SCR'].iloc[start:start+in_len].tolist()
            in_beam      = seq['unit1_pwr_60ghz'].iloc[start:start+in_len].tolist()
            out_beam_raw = seq['unit1_pwr_60ghz'].iloc[start+in_len:start+in_len+out_len].tolist()
            out_beam     = out_beam_raw + [np.nan] * (out_len - len(out_beam_raw))
            seq_idx      = seq['seq_index'].iloc[0:1].tolist()
            all_seqs.append(lidar + out_beam + in_beam + seq_idx)
            start += 1

    col_names = ([f'seq{i+1}'         for i in range(in_len)]  +
                 [f'Future_Beam{i+1}' for i in range(out_len)] +
                 [f'Beam{i+1}'        for i in range(in_len)]  +
                 ['seq_index'])

    all_seqs_df = pd.DataFrame(all_seqs, columns=col_names)

    #''' in SEQ. spliting 
    n = all_seq_idx.shape[0]
    train_idx = np.sort(all_seq_idx[:int(0.7 * n)])
    val_idx   = np.sort(all_seq_idx[int(0.7 * n):int(0.85 * n)])
    test_idx  = np.sort(all_seq_idx[int(0.85 * n):])
    #'''
    '''
    rng = np.random.default_rng(seed=seed)
    shuffled = rng.permutation(all_seq_idx)
    n = len(shuffled)
    train_idx = shuffled[:int(0.7  * n)]
    val_idx   = shuffled[int(0.7  * n):int(0.85 * n)]
    test_idx  = shuffled[int(0.85 * n):]
    #'''

    all_seqs_df[all_seqs_df['seq_index'].isin(train_idx)].to_csv(os.path.join(csv_dir, 'train_seqs.csv'), index=False)
    all_seqs_df[all_seqs_df['seq_index'].isin(val_idx  )].to_csv(os.path.join(csv_dir, 'val_seqs.csv'),   index=False)
    all_seqs_df[all_seqs_df['seq_index'].isin(test_idx )].to_csv(os.path.join(csv_dir, 'test_seqs.csv'),  index=False)
    print(f"[split seed={seed}]  train={len(train_idx)} seqs  val={len(val_idx)} seqs  test={len(test_idx)} seqs")

#%%
def make_loaders(portion, seq_length, csv_dir, batch_size=64):
    _kw = dict(num_workers=8, pin_memory=True, persistent_workers=True)
    train_loader = DataLoader(DataFeed(os.path.join(csv_dir, 'train_seqs.csv'), seq_length, portion=portion),
                              batch_size=batch_size, shuffle=True,  **_kw)
    val_loader   = DataLoader(DataFeed(os.path.join(csv_dir, 'val_seqs.csv'),   seq_length),
                              batch_size=128,       shuffle=False, **_kw)
    test_loader  = DataLoader(DataFeed(os.path.join(csv_dir, 'test_seqs.csv'),  seq_length),
                              batch_size=128,       shuffle=False, **_kw)
    return train_loader, val_loader, test_loader


#%% =================== Teacher Training ===================
def train_teacher(train_loader, val_loader, num_classes=64, num_epochs=10, patience=5):
    
    # set the teacher from the below 
    teacher = gru_cnn_teacher(num_classes).to(device) 
    #teacher = GruModel(num_classes).to(device) # from [10]


    optimizer = torch.optim.Adam(teacher.parameters(), lr=1e-3, weight_decay=1e-4)                     
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    criterion = nn.CrossEntropyLoss()
    best_val_loss = 1e10
    patience_counter = 0
    best_path = os.path.join(save_dir, "teacher.pth")

    print("Training TEACHER model...")
    for epoch in range(num_epochs):
        teacher.train()
        epoch_loss = 0.0
        for lidar_img, beam, label in train_loader:
            lidar_img = torch.swapaxes(lidar_img, 0, 1)                                      # [8, B, 216]
            lidar_img = torch.cat([lidar_img, torch.zeros_like(lidar_img[:3, ...])], dim=0)  # [11, B, 216]
            label = torch.cat([beam[..., -1:], label], dim=-1)
            label = torch.swapaxes(label, 0, 1)                                               # [4, B]
            lidar_img, label = lidar_img.to(device), label.to(device)
            h = teacher.initHidden(lidar_img.shape[1]).to(device)
            out, _ = teacher(lidar_img, h)
            loss = criterion(out[-4:, ...].reshape(-1, num_classes), label[-4:].flatten())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f}")

        # --- Validation & early stopping ---
        print(f"  [Val]", end=" ")
        top1, _, val_loss = evaluate_model(teacher, val_loader, num_classes, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(teacher.state_dict(), best_path)
            print("  --> best model saved")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1} (no val improvement for {patience} epochs)")
                break

    teacher.load_state_dict(torch.load(best_path))
    print("\n--- Evaluation TEACHER (best ckpt) ---")
    t_top1, t_top5, _ = evaluate_model(teacher, val_loader, num_classes, device)
    return teacher, t_top1, t_top5


#%% =================== Metadata Collection (for DF-KD) ===================
def collect_metadata(teacher, dataloader):
    teacher.eval()
    act_list = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[0].to(device)  # first tensor: inputs
            inputs = inputs.permute(1, 0, 2)  # (seq_len=8, batch, input_dim)
            inputs = torch.cat([inputs, torch.zeros_like(inputs[:3, ...])], dim=0)  # (11, batch, input_dim)
            batch_size = inputs.shape[1]

            h = teacher.initHidden(batch_size).to(device)
            _, features = teacher( inputs, h )  # features shape (batch_size, hidden_size)
            act_list.append(features)

    if len(act_list) == 0:
        raise RuntimeError("No metadata collected from teacher!")
    feats = torch.cat(act_list, dim=0)  # (total_samples, hidden_size)
    print(f"[Metadata] Collected {feats.shape[0]} features with dim {feats.shape[1]}")
    torch.save(feats, os.path.join(save_dir, 'metadata.pt'))
    return feats


#%% =================== Vanilla Student Training (no KD) ===================
def train_vanilla_student(train_loader, val_loader, num_classes=64, num_epochs=10, patience=5):

    student = student_model(num_classes).to(device)

    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    criterion = nn.CrossEntropyLoss()
    best_val_loss = 1e10
    patience_counter = 0
    best_path = os.path.join(save_dir, "vanilla_student.pth")

    print("\nTraining VANILLA STUDENT model (no KD)...")
    for epoch in range(num_epochs):
        student.train()
        epoch_loss = 0.0
        for lidar_img, beam, label in train_loader:
            lidar_img = torch.swapaxes(lidar_img, 0, 1)                                      # [8, B, 216]
            lidar_img = torch.cat([lidar_img, torch.zeros_like(lidar_img[:3, ...])], dim=0)  # [11, B, 216]
            label = torch.cat([beam[..., -1:], label], dim=-1)
            label = torch.swapaxes(label, 0, 1)                                               # [4, B]
            lidar_img, label = lidar_img.to(device), label.to(device)
            h = student.initHidden(lidar_img.shape[1])
            if h is not None: h = h.to(device)
            out, _ = student(lidar_img, h)
            loss = criterion(out[-4:, ...].reshape(-1, num_classes), label[-4:].flatten())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f}")

        print(f"  [Val]", end=" ")
        top1, _, val_loss = evaluate_model(student, val_loader, num_classes, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(student.state_dict(), best_path)
            print("  --> best model saved")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1} (no val improvement for {patience} epochs)")
                break

    student.load_state_dict(torch.load(best_path))
    print("\n--- Evaluation VANILLA STUDENT (best ckpt) ---")
    v_top1, v_top5, _ = evaluate_model(student, val_loader, num_classes, device)
    return student, v_top1, v_top5



#%% =================== KD Student Training ===================
def train_kd_student(teacher, train_loader, val_loader, num_classes=64, num_epochs=100, patience=5,
                     temp=2.0, alpha=0.6, seed=None):

    student_kd = student_model(num_classes).to(device)

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(student_kd.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    best_val_loss = 1e10
    patience_counter = 0
    best_path = os.path.join(save_dir, "kd_student.pth")

    print("\nTraining KD STUDENT model...")
    for epoch in range(num_epochs):
        student_kd.train()
        epoch_loss = 0.0
        for lidar_img, beam, label in train_loader:
            lidar_img = torch.swapaxes(lidar_img, 0, 1)
            lidar_img = torch.cat([lidar_img, torch.zeros_like(lidar_img[:3, ...])], dim=0)
            label = torch.cat([beam[..., -1:], label], dim=-1)
            label = torch.swapaxes(label, 0, 1)
            lidar_img, label = lidar_img.to(device), label[-4:].to(device)

            with torch.no_grad():
                h = teacher.initHidden(lidar_img.shape[1]).to(device)
                t_logits, _ = teacher(lidar_img, h)
                t_logits = t_logits[-4:].reshape(-1, num_classes)

            h = student_kd.initHidden(lidar_img.shape[1])
            if h is not None: h = h.to(device)
            s_logits, _ = student_kd(lidar_img, h)
            s_logits = s_logits[-4:].reshape(-1, num_classes)

            #--- 1. KD loss (CE + soft KL)
            ce = F.cross_entropy(s_logits, label.flatten())
            kl = F.kl_div(
                F.log_softmax(s_logits / temp, dim=1),
                F.softmax(t_logits / temp, dim=1),
                reduction='batchmean'
            ) * (temp ** 2)
            loss = (1 - alpha) * ce + alpha * kl

            #--- or 2. MSE loss:
            #loss = F.mse_loss(s_logits, t_logits)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f}")

        print(f"  [Val]", end=" ")
        top1, _, val_loss = evaluate_model(student_kd, val_loader, num_classes, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(student_kd.state_dict(), best_path)
            print("  --> best model saved")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1} (no val improvement for {patience} epochs)")
                break

    student_kd.load_state_dict(torch.load(best_path))
    print("\n--- Evaluation KD STUDENT (best ckpt) ---")
    kd_top1, kd_top5, _ = evaluate_model(student_kd, val_loader, num_classes, device)
    return student_kd, kd_top1, kd_top5


#%% =================== DF-KD Student Training ===================
def train_df_kd_student(teacher, val_loader, metadata, num_classes=64, 
                       num_epochs_g=500, num_epochs_s=500, num_samples=5000 ):

    z_fixed = torch.randn(num_samples, input_dim).to(device)

    generator = generator_model().to(device)
    #'''
    train_generator(generator, teacher, num_epochs=num_epochs_g, metadata=metadata)

    generator.eval()
    with torch.no_grad():
        fake_inputs_lidar = generator(z_fixed).permute(1, 0, 2)
    fake_inputs_fixed = torch.cat(
        [fake_inputs_lidar, torch.zeros_like(fake_inputs_lidar[:3, ...])], dim=0) 
    #'''    

    #fake_inputs_fixed = torch.randn( 11, num_samples, 216 ).to(device) # pure noise-based training (w/o generator)

    student_df = student_model(num_classes).to(device)

    train_student_fixed_data(fake_inputs_fixed, teacher, student_df, generator, num_epochs=num_epochs_s,
                             val_loader=val_loader, num_classes=num_classes)

    print("\n--- Evaluation STUDENT with DF-KD ---")
    df_top1, df_top5, _ = evaluate_model(student_df, val_loader, num_classes, device)
    return student_df, df_top1, df_top5




#%% =================== One Full Run ===================
def run_one( num_epochs_t=100, num_epochs_v=100, num_epochs_kdf=100,
            patience_t=5, patience_v=5, patience_kdf=5,
            portion=1., seed=42, seq_length=8, csv_dir='./utils',
            ):
    num_classes = 64

    # Re-generate train/val/test splits using this run's seed so each MC run
    # sees a genuinely different data partition (sequence-level, no leakage).
    generate_splits(seed, csv_dir, seq_length=seq_length)

    #''' Teacher tarining
    set_seed(seed)
    train_loader, val_loader, test_loader = make_loaders(portion, seq_length, csv_dir)
    teacher, _, _ = train_teacher(train_loader, val_loader, num_classes, num_epochs=num_epochs_t, patience=patience_t)
    print("\n--- Evaluation TEACHER (test set) ---")
    t_top1, t_top5, _ = evaluate_model(teacher, test_loader, num_classes, device)
    #'''
   


    ''' for the saved teacher model cases # this is train the teacher once and then use it to train other models for different settings
    set_seed(seed)
    seed_dir = f"./saved_models/teachers/seed_{seed}"
    teacher = gru_cnn_teacher(num_classes).to(device) 
    teacher.load_state_dict(torch.load(f"{seed_dir}/teacher.pth"))
    #'''



    #''' no KD: 1 ; if you dont want run this scenario, uncomment BUT comment the deafult values (which are for crash errors)
    set_seed(seed)
    train_loader, val_loader, test_loader = make_loaders(portion, seq_length, csv_dir) # NOTE MUST be called for identical data input for each scenario  
    vanilla_student, _, _ = train_vanilla_student(train_loader, val_loader, num_classes, num_epochs=num_epochs_v, patience=patience_v)
    print("\n--- Evaluation VANILLA STUDENT (test set) ---")
    v_top1, v_top5, _ = evaluate_model(vanilla_student, test_loader, num_classes, device)
    #'''
    
    #''' standard KD student 
    set_seed(seed)
    train_loader, val_loader, test_loader = make_loaders(portion, seq_length, csv_dir)
    kdf_student, _, _ = train_kd_student(teacher, train_loader, val_loader, num_classes, num_epochs=num_epochs_kdf, patience=patience_kdf, seed=seed)
    print("\n--- Evaluation KD STUDENT (test set) ---")
    kdf_top1, kdf_top5, _ = evaluate_model(kdf_student, test_loader, num_classes, device)
    #'''
    


    #''' df kd
    train_loader, val_loader, test_loader = make_loaders(portion, seq_length, csv_dir)
    metadata = collect_metadata(teacher, train_loader)
    
    df_student, _, _ = train_df_kd_student(teacher, val_loader, metadata, num_classes)
    print("\n--- Evaluation DF-KD STUDENT (test set) ---")
    df_top1, df_top5, _ = evaluate_model(df_student, test_loader, num_classes, device)
    #'''

    return t_top1, t_top5, v_top1, v_top5, kdf_top1, kdf_top5, df_top1, df_top5





#%% =================== Entry Point ===================
if __name__ == "__main__":

    import matplotlib
    import shutil
    matplotlib.use('Agg')

    run_id   = f"{time.strftime('%Y-%m-%d_%H_%M_%S')}_{os.getpid()}"
    save_dir = os.path.join(save_dir, run_id)
    os.makedirs(save_dir, exist_ok=True)
    csv_dir  = os.path.join("./csv_records", run_id)
    os.makedirs(csv_dir, exist_ok=True)
    print(f"Run directory: {save_dir}")
    scripts_dir = os.path.join(save_dir, "main_scripts")
    os.makedirs(scripts_dir, exist_ok=True)
    shutil.copy(__file__,                                                   os.path.join(scripts_dir, "main_teacher_KD.py"))
    shutil.copy(os.path.join(os.path.dirname(__file__), "main_DF_KD.py"),   os.path.join(scripts_dir, "main_DF_KD.py"))
    shutil.copy(os.path.join(os.path.dirname(__file__), "model.py"),        os.path.join(scripts_dir, "model.py"))
    shutil.copy(os.path.join(os.path.dirname(__file__), "loss_functions", "data_generator_lossess.py"),  os.path.join(scripts_dir, "data_generator_lossess.py"))
    shutil.copy(os.path.join(os.path.dirname(__file__), "loss_functions", "training_student_lossess.py"), os.path.join(scripts_dir, "training_student_lossess.py"))
    #%% --- main configs --- 
    num_classes = 64

    SEQ_LEN       = 8    # LiDAR frames fed to model

    NUM_EPOCHS_T   = 100; PATIENCE_T   = 5
    NUM_EPOCHS_V   = 100; PATIENCE_V   = 10
    NUM_EPOCHS_KDF = 100; PATIENCE_KDF = 10

    N_RUNS   = 20  # Monte Carlo runs

    rng   = np.random.default_rng()
    SEEDS = [2832, 1198, 8534, 6494, 8691, 4046, 3394, 1785, 9884, 2403, 4984, 464, 10, 3531, 8895, 6601, 8147, 5751, 1503, 835] # FIXED for the final results in r2
    #SEEDS = rng.integers(0, 10000, size=N_RUNS).tolist()
    print(f"\n Seeds: {SEEDS}")




    #%%---
    import utils.data_feed as _df_mod; scenario_num = _df_mod.scenario_num; data_root = _df_mod.data_root

    all_t_top1,   all_t_top5   = [], []
    all_v_top1,   all_v_top5   = [], []
    all_kdf_top1, all_kdf_top5 = [], []
    all_df_top1,  all_df_top5  = [], []

    for i, seed in enumerate(SEEDS):
        print(f"\n{'='*50}\n  RUN {i+1}/{N_RUNS}  (seed={seed})\n{'='*50}")

        t_top1, t_top5, v_top1, v_top5, kdf_top1, kdf_top5, df_top1, df_top5 = run_one(
            num_epochs_t=NUM_EPOCHS_T, num_epochs_v=NUM_EPOCHS_V,
            num_epochs_kdf=NUM_EPOCHS_KDF,
            patience_t=PATIENCE_T, patience_v=PATIENCE_V,
            patience_kdf=PATIENCE_KDF,
            portion=1.0, seed=seed, seq_length=SEQ_LEN, csv_dir=csv_dir,
            )

        all_t_top1.append(t_top1);     all_t_top5.append(t_top5)
        all_v_top1.append(v_top1);     all_v_top5.append(v_top5)
        all_kdf_top1.append(kdf_top1); all_kdf_top5.append(kdf_top5)
        all_df_top1.append(df_top1);   all_df_top5.append(df_top5)

    all_t_top1   = np.stack(all_t_top1)   # [N_RUNS, 4]
    all_t_top5   = np.stack(all_t_top5)
    all_v_top1   = np.stack(all_v_top1)
    all_v_top5   = np.stack(all_v_top5)
    all_kdf_top1 = np.stack(all_kdf_top1)
    all_kdf_top5 = np.stack(all_kdf_top5)
    all_df_top1  = np.stack(all_df_top1)
    all_df_top5  = np.stack(all_df_top5)

    def _report(label, top1, top5):
        n = top1.shape[0]
        print(f"\n  {label}")
        print(f"  Top-1  mean: {np.round(top1.mean(0),4)}  std: {np.round(top1.std(0),4)}  SEM: {np.round(top1.std(0)/np.sqrt(n),4)}")
        print(f"  Top-5  mean: {np.round(top5.mean(0),4)}  std: {np.round(top5.std(0),4)}  SEM: {np.round(top5.std(0)/np.sqrt(n),4)}")

    print(f"\n{'='*50}\n  FINAL RESULTS — {N_RUNS} MC runs\n  Seeds: {SEEDS}\n{'='*50}")
    _report("TEACHER",                  all_t_top1,   all_t_top5)
    _report("STUDENT (no KD)",          all_v_top1,   all_v_top5)
    _report("STUDENT (KD + features)",  all_kdf_top1, all_kdf_top5)
    _report("STUDENT (DF-KD)",          all_df_top1,  all_df_top5)

    np.save(os.path.join(save_dir, "mc_seeds.npy"),     np.array(SEEDS))
    np.save(os.path.join(save_dir, "mc_v_top1.npy"),    all_v_top1)
    np.save(os.path.join(save_dir, "mc_v_top5.npy"),    all_v_top5)
    np.save(os.path.join(save_dir, "mc_kdf_top1.npy"),  all_kdf_top1)
    np.save(os.path.join(save_dir, "mc_kdf_top5.npy"),  all_kdf_top5)
    np.save(os.path.join(save_dir, "mc_df_top1.npy"),   all_df_top1)
    np.save(os.path.join(save_dir, "mc_df_top5.npy"),   all_df_top5)
    print(f"\nResults saved to {save_dir}/")

    # -------- Plot --------
    slots = ['Current', '+1', '+2', '+3']
    x = np.arange(len(slots))
    t_mean1   = all_t_top1.mean(0)   * 100;  t_std1   = all_t_top1.std(0)   * 100
    t_mean5   = all_t_top5.mean(0)   * 100;  t_std5   = all_t_top5.std(0)   * 100
    v_mean1   = all_v_top1.mean(0)   * 100;  v_std1   = all_v_top1.std(0)   * 100
    v_mean5   = all_v_top5.mean(0)   * 100;  v_std5   = all_v_top5.std(0)   * 100
    kdf_mean1 = all_kdf_top1.mean(0) * 100;  kdf_std1 = all_kdf_top1.std(0) * 100
    kdf_mean5 = all_kdf_top5.mean(0) * 100;  kdf_std5 = all_kdf_top5.std(0) * 100
    df_mean1  = all_df_top1.mean(0)  * 100;  df_std1  = all_df_top1.std(0)  * 100
    df_mean5  = all_df_top5.mean(0)  * 100;  df_std5  = all_df_top5.std(0)  * 100

    def _make_fig(t_mean, t_std, v_mean, v_std, kdf_mean, kdf_std, df_mean, df_std, topk_label, y_lim):
        fig, ax = plt.subplots(figsize=(9, 4))
        offsets = [-0.15, -0.05, 0.05, 0.15]
        series = [
            (t_mean,   t_std,   'o-',  'steelblue',   'Teacher'),
            (v_mean,   v_std,   'D--', 'slategray',    'Student (no KD)'),
            (kdf_mean, kdf_std, 'P--', 'mediumpurple', 'Student (KD+feat)'),
            (df_mean,  df_std,  '^:',  'forestgreen',  'Student (DF-KD)'),
        ]
        for (mean, std, fmt, color, label), off in zip(series, offsets):
            ax.errorbar(x + off, mean, yerr=std, fmt=fmt, color=color,
                        linewidth=2, markersize=6, capsize=4, capthick=1.5, label=label)
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.12)
            for j, v in enumerate(mean):
                ax.text(j + off, mean[j] + std[j] + 0.8, f'{v:.1f}%',
                        ha='center', va='bottom', fontsize=7.5, color=color)
        ax.set_xticks(x); ax.set_xticklabels(slots)
        ax.set_xlabel('Prediction slot'); ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'{topk_label} beam prediction accuracy\n({N_RUNS} MC runs, error bars = ±std)')
        ax.set_ylim(*y_lim); ax.legend(loc='lower left', fontsize=8)
        ax.yaxis.grid(True, linestyle='--', alpha=0.5); ax.set_axisbelow(True)
        fig.tight_layout()
        return fig

    all1 = np.concatenate([t_mean1, v_mean1, kdf_mean1, df_mean1])
    all5 = np.concatenate([t_mean5, v_mean5, kdf_mean5, df_mean5])
    ylim1 = (all1.min() - 2, all1.max() + 2)
    ylim5 = (all5.min() - 2, all5.max() + 2)
    fig1 = _make_fig(t_mean1, t_std1, v_mean1, v_std1, kdf_mean1, kdf_std1, df_mean1, df_std1, 'Top-1', ylim1)
    fig5 = _make_fig(t_mean5, t_std5, v_mean5, v_std5, kdf_mean5, kdf_std5, df_mean5, df_std5, 'Top-5', ylim5)
    fig1.savefig(os.path.join(save_dir, "mc_top1.pdf"), dpi=150, bbox_inches='tight')
    fig5.savefig(os.path.join(save_dir, "mc_top5.pdf"), dpi=150, bbox_inches='tight')
    print(f"Plots saved to {save_dir}/mc_top1.pdf  and  mc_top5.pdf")

    import inspect as _inspect
    _t_tmp = gru_cnn_teacher(64)
    _s_tmp  = student_model(64)
    _sep = "=" * 60
    print(f"""
{_sep}
  EXPERIMENT CONFIGURATION
{_sep}
  Dataset
    Scenario   : {scenario_num}  ({data_root})
    Seq length : {SEQ_LEN}  (LiDAR frames fed to model)
    Num classes: 64 beams
    Train/Val/Test split regenerated per MC run (seed-dependent, 60/15/25 seq-level)

  Models
    Teacher : {type(_t_tmp).__name__} — {(f"embed={_t_tmp.embed.out_features}, GRU hidden={_t_tmp.hidden_size}, layers={_t_tmp.num_layers}, dropout={_t_tmp.dropout.p}") if hasattr(_t_tmp, 'embed') else (f"embed={_t_tmp.fc_embed.out_features}, GRU hidden={_t_tmp.hidden_size}, layers={_t_tmp.num_layers}, dropout={_t_tmp.dropout.p}")}
    Student : {type(_s_tmp).__name__} — embed={_s_tmp.embed.out_features}, GRU hidden={_s_tmp.hidden_size}, layers={_s_tmp.num_layers}, dropout={_s_tmp.dropout1.p if hasattr(_s_tmp, "dropout1") else _s_tmp.dropout.p}

  Training — Teacher & Vanilla Student
    Optimiser  : Adam  lr=1e-3  weight_decay=1e-4
    Max epochs : {NUM_EPOCHS_T} / {NUM_EPOCHS_V}  (teacher / vanilla student)
    Patience   : {PATIENCE_T} / {PATIENCE_V} epochs (early stopping on val Top-1)
    Batch size : 64 (train) / 128 (val/test)

  Monte Carlo
    N_RUNS     : {N_RUNS}
    Seeds      : {SEEDS}
{_sep}
""")
    del _inspect, _sep, _t_tmp, _s_tmp
