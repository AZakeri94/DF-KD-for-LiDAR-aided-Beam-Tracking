import numpy as np
import pandas as pd

scenario_num  = 8
data_root = f'/projappl/project_2013517/DeepSense/datasets/Scenario{scenario_num}/scenario{scenario_num}.csv'
#data_root = '/projappl/project_2013517/DeepSense/datasets/scenario9_dev/scenario9.csv'
in_len = 8
out_len = 3 # No. of future beams to predict given time slot

all_data = pd.read_csv(data_root)
all_seq_idx = all_data['seq_index'].unique()

all_seq_split = []

for i in all_seq_idx:
    tmp = all_data[all_data['seq_index'] == i]
    tmp = tmp[['unit1_pwr_60ghz', 'unit1_lidar_SCR', 'seq_index']]
    all_seq_split.append(tmp)

all_seqs = []
for seq in all_seq_split:
    start = 0
    while start + in_len <= seq.shape[0]:
        lidar = seq['unit1_lidar_SCR'].iloc[start:start+in_len].tolist()
        in_beam = seq['unit1_pwr_60ghz'].iloc[start:start+in_len].tolist()
        out_beam_raw = seq['unit1_pwr_60ghz'].iloc[start+in_len:start+in_len+out_len].tolist()
        out_beam = out_beam_raw + [np.nan] * (out_len - len(out_beam_raw))  # pad missing future beams
        seq_idx = seq['seq_index'].iloc[0:1].tolist()
        all_seqs.append(lidar + out_beam + in_beam + seq_idx)
        start += 1

#col_names = ['seq1', 'seq2', 'seq3', 'seq4', 'seq5', 'seq6', 'seq7', 'seq8'] + ['Future_Beam1', 'Future_Beam2', 'Future_Beam3'] + ['Beam1', 'Beam2', 'Beam3', 'Beam4', 'Beam5', 'Beam6', 'Beam7', 'Beam8'] + ['seq_index']
col_names = (
    [f'seq{i+1}' for i in range(in_len)] +
    [f'Future_Beam{i+1}' for i in range(out_len)] +
    [f'Beam{i+1}' for i in range(in_len)] +
    ['seq_index']
)

all_seqs = pd.DataFrame(all_seqs, columns = col_names)

rng = np.random.default_rng(seed=42)
shuffled_seq_idx = rng.permutation(all_seq_idx)

n = shuffled_seq_idx.shape[0]
train_seq_idx = shuffled_seq_idx[:int(0.6 * n)]
val_seq_idx   = shuffled_seq_idx[int(0.6 * n):int(0.75 * n)]
test_seq_idx  = shuffled_seq_idx[int(0.75 * n):]

train_seqs = all_seqs[all_seqs['seq_index'].isin(train_seq_idx)]
val_seqs   = all_seqs[all_seqs['seq_index'].isin(val_seq_idx)]
test_seqs  = all_seqs[all_seqs['seq_index'].isin(test_seq_idx)]

train_seqs.to_csv('./utils/train_seqs.csv', index=False)
val_seqs.to_csv('./utils/val_seqs.csv',     index=False)
test_seqs.to_csv('./utils/test_seqs.csv',   index=False)

print(f'Train segments: {len(train_seq_idx)} ({len(train_seqs)} rows)')
print(f'Val   segments: {len(val_seq_idx)}   ({len(val_seqs)} rows)')
print(f'Test  segments: {len(test_seq_idx)}  ({len(test_seqs)} rows)')
