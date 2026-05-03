import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, classification_report,
    precision_score, recall_score, f1_score
)
import networkx as nx
from node2vec import Node2Vec
import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
from sklearn.decomposition import PCA
import os
import math
import torch.nn.functional as F

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.cluster._spectral")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.decomposition")

# ----------------------------------------------------------------------------
# 0) Reproducibility
# ----------------------------------------------------------------------------
random_seed = 46
SEED = random_seed
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------------------------------------------------------------------
# 1) Data Loading
# ----------------------------------------------------------------------------
df = pd.read_csv('data.csv')
Data_MT_HCC_Ver2 = df.copy()

# ----------------------------------------------------------------------------
# 2) Targets
# ----------------------------------------------------------------------------
y_main_df = Data_MT_HCC_Ver2[['ASCT', 'VART', 'ENCET', 'HEPT']]
# Aux: ONLY DEADT
y_aux_df  = Data_MT_HCC_Ver2[['DEADT']].copy()

# ----------------------------------------------------------------------------
# 3) Disease list (10)
# ----------------------------------------------------------------------------
disease_list = [
    'HBV','HCV','HIV','UVH',
    'DIABETES_CC','DIABETES_NO_CC',
    'MLD','MA','HEMOCHROMATOSIS',
    'MST'
]

# ----------------------------------------------------------------------------
# 4) Build X
# ----------------------------------------------------------------------------
features_to_drop = [
    'ASCT','VART','ENCET','HEPT',   
    'DEADT',                
    'OTHER'                        
]
features_to_drop = [c for c in features_to_drop if c in Data_MT_HCC_Ver2.columns]
X_all = Data_MT_HCC_Ver2.drop(columns=features_to_drop, errors='ignore')

# ----------------------------------------------------------------------------
# 5) Train-Test Split
# ----------------------------------------------------------------------------
X_train_df, X_test_df, y_train_main_df, y_test_main_df, y_train_aux_df, y_test_aux_df = train_test_split(
    X_all, y_main_df, y_aux_df,
    test_size=0.2,
    stratify=y_main_df,
    random_state=SEED
)

y_train_deadt = y_train_aux_df['DEADT'].values
y_test_deadt  = y_test_aux_df['DEADT'].values

train_rows_df = df.loc[X_train_df.index].copy()
test_rows_df  = df.loc[X_test_df.index].copy()

# ----------------------------------------------------------------------------
# 6) IF Model 
# ----------------------------------------------------------------------------
GROUP_ORDER = [
    'Male_0_40_O0','Male_40_60_O0','Male_60+_O0',
    'Female_0_40_O0','Female_40_60_O0','Female_60+_O0',
    'Male_0_40_O1','Male_40_60_O1','Male_60+_O1',
    'Female_0_40_O1','Female_40_60_O1','Female_60+_O1'
]
NUM_GROUPS = len(GROUP_ORDER)  # 12

def age_band_from_age(age):
    if age <= 40:
        return '0_40'
    elif age <= 60:
        return '40_60'
    else:
        return '60+'

def group_name_age_sex_other(row):
    is_male = (row['ID_SEX'] == 1)
    band = age_band_from_age(row['AGE'])
    other_val = int(row.get('OTHER', 0))  # default to 0 if absent (defensive)
    return ('Male_' if is_male else 'Female_') + band + f"_O{other_val}"

def group_index_age_sex_other(row):
    name = group_name_age_sex_other(row)
    return GROUP_ORDER.index(name)

class LearnableIFModel(nn.Module):
    def __init__(self, disease_list, num_targets, num_groups):
        super().__init__()
        self.disease_list = disease_list
        self.num_targets  = num_targets
        self.num_groups   = num_groups

        torch.manual_seed(random_seed)
        self.if_scores = nn.Parameter(  # (10,4)
            torch.rand(len(disease_list), num_targets, device=device, requires_grad=True)
        )
        self.group_multipliers = nn.Parameter(  # (10, num_groups)
            torch.rand(len(disease_list), num_groups, device=device, requires_grad=True)
        )

    def forward(self, X_diseases, group_indices):
        # X_diseases: (N,10)  group_indices: (N,)
        group_mults = self.group_multipliers[:, group_indices]  # (10,N)
        group_mults = group_mults.T                              # (N,10)
        weighted_disease = X_diseases * group_mults              # (N,10)
        weighted_scores  = torch.matmul(weighted_disease, self.if_scores) # (N,4)
        return weighted_scores

if_model = LearnableIFModel(disease_list, num_targets=4, num_groups=NUM_GROUPS).to(device)

def get_group_indices_age_sex_other(X_rows_df):
    idxs = []
    for _, r in X_rows_df.iterrows():
        idxs.append(group_index_age_sex_other(r))
    return np.array(idxs, dtype=np.int64)

# Prepare tensors for IF training (train split only)
X_train_diseases_np = train_rows_df[disease_list].values.astype(np.float32)
X_train_diseases_t  = torch.tensor(X_train_diseases_np, dtype=torch.float32, device=device)
y_train_main_t      = torch.tensor(y_train_main_df.values, dtype=torch.float32, device=device)

group_indices_train = get_group_indices_age_sex_other(train_rows_df)
group_indices_train_t = torch.tensor(group_indices_train, dtype=torch.long, device=device)

# Train IF model (simple MSE to fit main labels)
if_optimizer = optim.Adam(if_model.parameters(), lr=0.002)
if_criterion = nn.MSELoss()
num_epochs_if = 1000

for ep in range(num_epochs_if):
    if_model.train()
    if_optimizer.zero_grad()
    weighted_scores = if_model(X_train_diseases_t, group_indices_train_t)
    loss_if = if_criterion(weighted_scores, y_train_main_t)
    loss_if.backward()
    if_optimizer.step()
    if (ep+1) % 50 == 0:
        print(f"[IF Model] Ep {ep+1}/{num_epochs_if}, Loss={loss_if.item():.4f}")

if_model.eval()

# ----------------------------------------------------------------------------
# 7) Graph Building (Node2Vec) with Gender×Age×OTHER groups — TRAIN ONLY
# ----------------------------------------------------------------------------
def calc_cond_impact_score(d1, d2, t_idx, group_data):
    d1_idx = disease_list.index(d1)
    d2_idx = disease_list.index(d2)
    IF1 = if_model.if_scores[d1_idx, t_idx]
    IF2 = if_model.if_scores[d2_idx, t_idx]
    co_occ = group_data[(group_data[d1]==1) & (group_data[d2]==1)].shape[0]
    alpha = co_occ / group_data.shape[0] if group_data.shape[0] > 0 else 0.0
    return (abs(IF1 * IF2 * alpha)).item()

def split_groups_age_sex_other(rows_df):
    buckets = {name: [] for name in GROUP_ORDER}
    for idx, row in rows_df.iterrows():
        gname = group_name_age_sex_other(row)
        buckets[gname].append(idx)
    out = {name: rows_df.loc[id_list] if len(id_list)>0 else rows_df.iloc[0:0] for name, id_list in buckets.items()}
    return out  # dict of 12 DataFrames (some may be empty)

def create_representative_graph(group_data):
    G = nx.DiGraph()
    if group_data.shape[0] == 0:
        return G
    for d in disease_list:
        G.add_node(d)
    for d1 in disease_list:
        for d2 in disease_list:
            if d1 != d2:
                w = 0.0
                for t_idx in range(4):
                    w += calc_cond_impact_score(d1, d2, t_idx, group_data)
                if w > 0:
                    G.add_edge(d1, d2, weight=w)
    return G

def extract_group_emb_32(rows_df_train):
    groups = split_groups_age_sex_other(rows_df_train)
    all_emb = {}
    for gname in GROUP_ORDER:
        gdata = groups[gname]
        if gdata.shape[0] == 0:
            all_emb[gname] = {d: np.zeros(32, dtype=np.float32) for d in disease_list}
            continue
        G = create_representative_graph(gdata)
        if len(G.edges) > 0:
            node2vec = Node2Vec(
                G, dimensions=32, walk_length=50, num_walks=100,
                workers=1, quiet=True, weight_key='weight', seed=random_seed
            )
            n2v_model = node2vec.fit(window=3, min_count=1)
            group_emb = {node: n2v_model.wv[node] for node in G.nodes}
            # ensure every disease exists
            for d in disease_list:
                if d not in group_emb:
                    group_emb[d] = np.zeros(32, dtype=np.float32)
        else:
            group_emb = {d: np.zeros(32, dtype=np.float32) for d in disease_list}
        all_emb[gname] = group_emb
    return all_emb

group_emb_32 = extract_group_emb_32(train_rows_df)

# Initialize learnable group embeddings from Node2Vec (via PCA on mean(10×32))
group_embedding_rows = []
for gname in GROUP_ORDER:
    disease_emb_array = np.stack([group_emb_32[gname][d] for d in disease_list], axis=0)  # (10,32)
    group_emb_mean = disease_emb_array.mean(axis=0)  # (32,)
    group_embedding_rows.append(group_emb_mean)

group_embedding_np = np.stack(group_embedding_rows, axis=0)  # shape: (NUM_GROUPS, 32)
GROUP_EMB_DIM = min(8, group_embedding_np.shape[0], group_embedding_np.shape[1])  # up to 8
pca = PCA(n_components=GROUP_EMB_DIM, random_state=random_seed)
group_embeddings_pca = pca.fit_transform(group_embedding_np)  # (NUM_GROUPS, GROUP_EMB_DIM)
group_embeddings_pca_t = torch.tensor(group_embeddings_pca, dtype=torch.float32, device=device)

# ----------------------------------------------------------------------------
# 8) EnhancedMultiTaskNet — single aux (DEADT)
# ----------------------------------------------------------------------------
class EnhancedMultiTaskNet(nn.Module):
    def __init__(self, input_dim, num_heads=2, num_groups=12, group_emb_dim=8, num_diseases=10, disease_emb_dim=32):
        super().__init__()
        self.input_dim       = input_dim
        self.num_heads       = num_heads
        self.num_groups      = num_groups
        self.group_emb_dim   = group_emb_dim
        self.num_diseases    = num_diseases
        self.disease_emb_dim = disease_emb_dim

        # Learnable group embeddings (init later from PCA)
        self.group_embeddings = nn.Parameter(
            torch.rand(num_groups, group_emb_dim, device=device, requires_grad=True)
        )

        # Shared tabular tower
        self.shared_fc1 = nn.Linear(input_dim, 256)
        self.shared_fc2 = nn.Linear(256, 128)
        self.shared_fc3 = nn.Linear(128, 64)
        self.dropout    = nn.Dropout(0.1)
        self.relu       = nn.ReLU()

        # Disease embedding conv stack (disease_emb_dim + group_emb_dim channels)
        in_ch = self.disease_emb_dim + self.group_emb_dim
        self.conv1 = nn.Conv1d(in_channels=in_ch, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32,   out_channels=64, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool1d(kernel_size=2)  # 10 -> 5

        self.flatten = nn.Flatten()
        self.fc_diseases = nn.Linear(64*5, 256)
        self.fc_diseases_relu = nn.ReLU()

        # Attention over concatenated [shared(64) + disease(256)] = 320
        self.multihead_attn = nn.MultiheadAttention(embed_dim=320, num_heads=num_heads, batch_first=True)
        self.post_attn_fc   = nn.Linear(320, 64)
        self.post_attn_relu = nn.ReLU()

        # Aux head: DEADT
        self.deadt_fc = nn.Linear(64, 1)

        # Main heads consume attended(64) + 1 aux prob = 65
        self.main_fc1 = nn.Linear(64 + 1, 192)
        self.main_fc2 = nn.Linear(192, 64)
        self.main_fc3 = nn.Linear(64, 4)  # ASCT, VART, ENCET, HEPT

    def forward(self, x, node2vec_32, group_idx):
        # Shared tabular path
        shared = self.relu(self.shared_fc1(x))
        shared = self.relu(self.shared_fc2(shared))
        shared = self.dropout(self.relu(self.shared_fc3(shared)))  # (N,64)

        # Expand group embedding per disease
        group_emb = self.group_embeddings[group_idx]                              # (N, group_emb_dim)
        group_emb_expanded = group_emb.unsqueeze(1).repeat(1, self.num_diseases, 1)  # (N,10,group_emb_dim)

        # Combine disease Node2Vec (N,10,32) + group (N,10,group_emb_dim)
        disease_combined = torch.cat([node2vec_32, group_emb_expanded], dim=2)   # (N,10,32+group_emb_dim)
        conv_input = disease_combined.permute(0, 2, 1)                            # (N,32+group_emb_dim,10)

        z = self.relu(self.conv1(conv_input))                                     # (N,32,10)
        z = self.relu(self.conv2(z))                                              # (N,64,10)
        z = self.pool(z)                                                          # (N,64,5)

        z = self.flatten(z)                                                       # (N,320)
        z = self.fc_diseases_relu(self.fc_diseases(z))                            # (N,256)

        combined = torch.cat([shared, z], dim=1).unsqueeze(1)                     # (N,1,320)
        attn_out, _ = self.multihead_attn(combined, combined, combined)           # (N,1,320)
        attn_out = attn_out.squeeze(1)                                            # (N,320)
        attended = self.post_attn_relu(self.post_attn_fc(attn_out))               # (N,64)

        # Aux prediction
        deadt_out = self.deadt_fc(attended).squeeze(-1)                           # (N,)

        # Feed ONLY predicted DEADT prob into main
        dprob = torch.sigmoid(deadt_out).unsqueeze(1)                             # (N,1)
        main_in = torch.cat([attended, dprob], dim=1)                             # (N,65)

        main_z = self.relu(self.main_fc1(main_in))
        main_z = self.relu(self.main_fc2(main_z))
        out4   = self.main_fc3(main_z)                                            # (N,4)

        asct_out  = out4[:, 0]
        vart_out  = out4[:, 1]
        encet_out = out4[:, 2]
        hept_out  = out4[:, 3]

        return asct_out, vart_out, encet_out, hept_out, deadt_out

# Use PCA-determined dimension
GROUP_EMB_DIM = group_embeddings_pca.shape[1]

model = EnhancedMultiTaskNet(
    input_dim=X_train_df.shape[1],
    num_heads=2,
    num_groups=NUM_GROUPS,
    group_emb_dim=GROUP_EMB_DIM,
    num_diseases=len(disease_list),
    disease_emb_dim=32
).to(device)

with torch.no_grad():
    model.group_embeddings.copy_(group_embeddings_pca_t)
    print(f"Initialized group embeddings with Node2Vec-PCA (groups={NUM_GROUPS}, dim={GROUP_EMB_DIM}).")

# ----------------------------------------------------------------------------
# 9) Build Patient Node2Vec Embeddings (per patient) using 12-group mapping
# ----------------------------------------------------------------------------
def build_patient_embeddings(rows_df, all_group_emb_32):
    # Normalized *_DATE (non-leaky timing features used only to scale per-disease embeddings)
    disease_date_cols = [f"{d}_DATE" for d in disease_list]
    normed = rows_df[disease_date_cols].copy()
    for c in disease_date_cols:
        if c not in normed.columns:
            normed[c] = 0.0
        meanv = normed[c].mean()
        stdv  = normed[c].std() + 1e-6
        normed[c] = (normed[c] - meanv) / stdv

    patient_node2vec_emb = []
    for idx, row in tqdm(rows_df.iterrows(), total=rows_df.shape[0], desc="Building patient embeddings"):
        gname = group_name_age_sex_other(row)
        group_emb_dict = all_group_emb_32.get(gname, {d: np.zeros(32, dtype=np.float32) for d in disease_list})

        row_emb = []
        for d in disease_list:
            if row.get(d, 0) == 1:
                dur_val = normed.at[idx, f"{d}_DATE"] if f"{d}_DATE" in normed.columns else 0.0
                emb = group_emb_dict.get(d, np.zeros(32, dtype=np.float32))
                row_emb.append(emb * dur_val)
            else:
                row_emb.append(np.zeros(32, dtype=np.float32))
        row_emb = np.array(row_emb, dtype=np.float32)  # (10,32)
        patient_node2vec_emb.append(row_emb)

    patient_node2vec_emb = np.stack(patient_node2vec_emb, axis=0)  # (N,10,32)
    return patient_node2vec_emb

train_node2vec_emb = build_patient_embeddings(train_rows_df, group_emb_32)
test_node2vec_emb  = build_patient_embeddings(test_rows_df,  group_emb_32)

train_node2vec_emb_t = torch.tensor(train_node2vec_emb, dtype=torch.float32, device=device)
test_node2vec_emb_t  = torch.tensor(test_node2vec_emb,  dtype=torch.float32, device=device)

# Group indices for model input (need OTHER here; we read from rows_df, not X)
train_gidx_t = torch.tensor(get_group_indices_age_sex_other(train_rows_df), dtype=torch.long, device=device)
test_gidx_t  = torch.tensor(get_group_indices_age_sex_other(test_rows_df),  dtype=torch.long, device=device)

# ----------------------------------------------------------------------------
# 10) Datasets & Loaders
# ----------------------------------------------------------------------------
train_dataset = TensorDataset(
    torch.tensor(X_train_df.values, dtype=torch.float32, device=device),
    train_node2vec_emb_t,
    train_gidx_t,
    torch.tensor(y_train_main_df.values, dtype=torch.float32, device=device),  # (N,4)
    torch.tensor(y_train_deadt, dtype=torch.float32, device=device).view(-1)   # (N,)
)
test_dataset = TensorDataset(
    torch.tensor(X_test_df.values, dtype=torch.float32, device=device),
    test_node2vec_emb_t,
    test_gidx_t,
    torch.tensor(y_test_main_df.values, dtype=torch.float32, device=device),
    torch.tensor(y_test_deadt, dtype=torch.float32, device=device).view(-1)
)

batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

# ----------------------------------------------------------------------------
# 11) Losses / Optimizer
# ----------------------------------------------------------------------------
asct_pos_weight  = torch.tensor([1.0],   dtype=torch.float32, device=device)
vart_pos_weight  = torch.tensor([1.0],   dtype=torch.float32, device=device)
encet_pos_weight = torch.tensor([1.0],   dtype=torch.float32, device=device)
hept_pos_weight  = torch.tensor([10.0], dtype=torch.float32, device=device)  # tuned

criterion_asct  = nn.BCEWithLogitsLoss(reduction='none', pos_weight=asct_pos_weight)
criterion_vart  = nn.BCEWithLogitsLoss(reduction='none', pos_weight=vart_pos_weight)
criterion_encet = nn.BCEWithLogitsLoss(reduction='none', pos_weight=encet_pos_weight)
criterion_hept  = nn.BCEWithLogitsLoss(reduction='none', pos_weight=hept_pos_weight)
criterion_deadt = nn.BCEWithLogitsLoss(reduction='none')

optimizer = optim.AdamW(model.parameters(), lr=0.00021761424732883586, weight_decay=0.0)
print('Using tuned hyperparameters: lr=0.00021761424732883586, batch_size=128, num_heads=2, main_fc1_dim=192, loss_weight_deadt=1.0, hept_pos_weight=40.0, dropout=0.1, weight_decay=0.0')

IF_SCALING_FACTOR = 0.5
loss_weights = {'asct':1.0,'vart':1.0,'encet':1.0,'hept':1.0,'deadt':1.0}

def compute_if_weights(inputs, if_scores_tensor, disease_list):
    original_dim = X_train_df.shape[1]
    disease_cols_idx = []
    for d in disease_list:
        if d in X_train_df.columns:
            disease_cols_idx.append(list(X_train_df.columns).index(d))
    if len(disease_cols_idx) == 0:
        return torch.ones(inputs.size(0), 4, device=device)
    disease_data = inputs[:, :original_dim][:, disease_cols_idx]             # (N,10)
    patient_if_sum = torch.matmul(disease_data, if_scores_tensor[:len(disease_cols_idx), :])  # (N,4)
    return 1 + IF_SCALING_FACTOR * patient_if_sum

# ----------------------------------------------------------------------------
# 12) Training Loop
# ----------------------------------------------------------------------------
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    total_asct_loss=total_vart_loss=total_encet_loss=total_hept_loss=0.0
    total_deadt_loss=0.0

    for (inputs, node2vec_32, group_idx, labels_main, labels_deadt) in train_loader:
        optimizer.zero_grad()
        asct_out, vart_out, encet_out, hept_out, deadt_out = model(inputs, node2vec_32, group_idx)

        labels_asct  = labels_main[:,0]
        labels_vart  = labels_main[:,1]
        labels_encet = labels_main[:,2]
        labels_hept  = labels_main[:,3]

        if_weights = compute_if_weights(inputs, if_model.if_scores, disease_list)

        la_vals = criterion_asct(asct_out, labels_asct)
        lv_vals = criterion_vart(vart_out, labels_vart)
        le_vals = criterion_encet(encet_out, labels_encet)
        lh_vals = criterion_hept(hept_out, labels_hept)

        la = (la_vals * if_weights[:,0]).mean()
        lv = (lv_vals * if_weights[:,1]).mean()
        le = (le_vals * if_weights[:,2]).mean()
        lh = (lh_vals * if_weights[:,3]).mean()

        ld_vals = criterion_deadt(deadt_out, labels_deadt)
        ld = ld_vals.mean()

        total_loss = (
            loss_weights['asct']*la +
            loss_weights['vart']*lv +
            loss_weights['encet']*le +
            loss_weights['hept']*lh +
            loss_weights['deadt']*ld
        )
        total_loss.backward()
        optimizer.step()

        total_asct_loss  += la.item()
        total_vart_loss  += lv.item()
        total_encet_loss += le.item()
        total_hept_loss  += lh.item()
        total_deadt_loss += ld.item()

    print(
        f"Epoch [{epoch+1}/{num_epochs}] => "
        f"ASCT={total_asct_loss:.4f}, VART={total_vart_loss:.4f}, "
        f"ENCET={total_encet_loss:.4f}, HEPT={total_hept_loss:.4f}, "
        f"DEADT={total_deadt_loss:.4f}"
    )

# ----------------------------------------------------------------------------
# 13) Evaluation
# ----------------------------------------------------------------------------
model.eval()
with torch.no_grad():
    asct_preds,asct_trues=[],[]
    vart_preds,vart_trues=[],[]
    encet_preds,encet_trues=[],[]
    hept_preds,hept_trues=[],[]
    deadt_preds,deadt_trues=[],[]

    for (inputs, node2vec_32, group_idx, labels_main, labels_deadt) in test_loader:
        asct_out, vart_out, encet_out, hept_out, deadt_out = model(inputs, node2vec_32, group_idx)

        asct_prob  = torch.sigmoid(asct_out).cpu().numpy()
        vart_prob  = torch.sigmoid(vart_out).cpu().numpy()
        encet_prob = torch.sigmoid(encet_out).cpu().numpy()
        hept_prob  = torch.sigmoid(hept_out).cpu().numpy()
        deadt_prob = torch.sigmoid(deadt_out).cpu().numpy()

        asct_preds.extend(asct_prob);  asct_trues.extend(labels_main[:,0].cpu().numpy())
        vart_preds.extend(vart_prob);  vart_trues.extend(labels_main[:,1].cpu().numpy())
        encet_preds.extend(encet_prob);encet_trues.extend(labels_main[:,2].cpu().numpy())
        hept_preds.extend(hept_prob);  hept_trues.extend(labels_main[:,3].cpu().numpy())

        deadt_preds.extend(deadt_prob);deadt_trues.extend(labels_deadt.cpu().numpy())

thresholds = {
    'ASCT':0.5, 'VART':0.5, 'ENCET':0.5, 'HEPT':0.85, 'DEADT':0.5
}

def evaluate_task(task_name, preds, trues):
    preds = np.array(preds); trues = np.array(trues)
    if np.unique(trues).size > 1:
        auc_score = roc_auc_score(trues, preds)
        print(f"\nAUC for {task_name}: {auc_score:.4f}")
        thr = thresholds.get(task_name, 0.5)
        yhat = (preds > thr).astype(int)
        print(f"Classification Report ({task_name}, threshold={thr}):\n{classification_report(trues, yhat)}")

        # Threshold sweep
        t_range = np.arange(0, 1.05, 0.05)
        macro_ps, macro_rs, macro_fs = [], [], []
        for t in t_range:
            yhat_t = (preds > t).astype(int)
            mp = precision_score(trues, yhat_t, average='macro', zero_division=0)
            mr = recall_score(trues, yhat_t, average='macro', zero_division=0)
            mf = f1_score(trues, yhat_t, average='macro', zero_division=0)
            macro_ps.append(mp); macro_rs.append(mr); macro_fs.append(mf)

        return auc_score
    else:
        print(f"\nInsufficient data for {task_name} (only one class present).")
        return None

# Evaluate
asct_auc  = evaluate_task('ASCT',  asct_preds,  asct_trues)
vart_auc  = evaluate_task('VART',  vart_preds,  vart_trues)
encet_auc = evaluate_task('ENCET', encet_preds, encet_trues)
hept_auc  = evaluate_task('HEPT',  hept_preds,  hept_trues)

# Evaluate aux
evaluate_task('DEADT', deadt_preds, deadt_trues)

# Average AUC across mains
main_aucs = [a for a in [asct_auc, vart_auc, encet_auc, hept_auc] if a is not None]
if len(main_aucs) > 0:
    print(f"\nAverage AUC across ASCT, VART, ENCET, HEPT: {sum(main_aucs)/len(main_aucs):.4f}")
else:
    print("\nNot enough data to compute average AUC across main tasks.")

import re
import numpy as np, pandas as pd, matplotlib.pyplot as plt, networkx as nx, shap
import torch
from collections import Counter

# ===== Saving config
SAVE_PNG = True
PNG_DPI  = 220
OUT_DIR  = "patient_figs"
os.makedirs(OUT_DIR, exist_ok=True)

_required = [
    'model','device','if_model','disease_list','train_rows_df','test_rows_df',
    'X_train_df','X_test_df','y_test_main_df',
    'train_node2vec_emb_t','test_node2vec_emb_t','train_gidx_t','test_gidx_t',
    'group_name_age_sex_other','group_index_age_sex_other',
    'split_groups_age_sex_other',
    'y_train_main_df','y_train_aux_df','y_test_aux_df'
]
for _v in _required:
    if _v not in globals():
        raise RuntimeError(f"Missing '{_v}'. Please run the main training/eval cell first.")

# ===== Configs
HEADS = ['asct','vart','encet','hept']
HEAD_TITLES = {'asct':'ASCT','vart':'VART','encet':'ENCET','hept':'HEPT'}
THRESHOLDS = {'ASCT':0.5,'VART':0.5,'ENCET':0.5,'HEPT':0.85}
RANDOM_PERSON = False    # choose a random test patient
PERSON_POS = 303         # positional index into X_test_df if RANDOM_PERSON=False
TOPK_SHAP = 15
BG_SIZE = 200
NSAMPLES_SHAP = 200
LABEL_TOP_GROUP = 30  
LABEL_TOP_PERSON = 20

# ===== Chronocity mapping (standalone)
CHRON_DATE_MAP = {
    'HBV':'HBV_DATE','HCV':'HCV_DATE','HIV':'HIV_DATE','UVH':'UVH_DATE',
    'DIABETES_CC':'DIABETES_CC_DATE','DIABETES_NO_CC':'DIABETES_NO_CC_DATE',
    'MLD':'MLD_DATE','MA':'MA_DATE','HEMOCHROMATOSIS':'HEMOCHROMATOSIS_DATE','MST':'MST_DATE'
}
date_stats = {}
for d, col in CHRON_DATE_MAP.items():
    if col in train_rows_df.columns:
        m = float(train_rows_df[col].mean()); s = float(train_rows_df[col].std() + 1e-6)
        date_stats[col] = (m, s)

# ===== Feature name display helper (SHAP only)
_date_suffix_pat = re.compile(r'(_DATE|_Date)$')
def _pretty_feat_name(name: str) -> str:
    # Only change display name (do NOT touch model inputs)
    return _date_suffix_pat.sub('_Duration', name)

# ===== Helpers
TARGETS = ['ASCT','VART','ENCET','HEPT']
t_idx_map = {t:i for i,t in enumerate(TARGETS)}

def _chron_factor_from_row(row, disease):
    if disease not in row.index or row[disease]!=1: return 0.0
    col = CHRON_DATE_MAP.get(disease, None)
    if (col is None) or (col not in row.index) or (col not in date_stats): return 0.5
    mu, sd = date_stats[col]; z = (row[col] - mu) / (sd + 1e-6)
    return float(1.0 / (1.0 + np.exp(-1.0*z)))

def _co_occ_alpha(group_df, d1, d2):
    if group_df.shape[0] == 0: return 0.0
    return group_df[(group_df[d1]==1) & (group_df[d2]==1)].shape[0] / group_df.shape[0]

def _get_IF_adj_for_group(group_idx):
    with torch.no_grad():
        IF_base = if_model.if_scores.detach().cpu().numpy()                     # (10,4)
        gm = if_model.group_multipliers[:, group_idx].detach().cpu().numpy()    # (10,)
    return IF_base * gm[:, None]                                                # (10,4)

def build_group_graph_with_IFadj(group_df, IF_adj):
    G = nx.DiGraph()
    for d in disease_list: G.add_node(d)
    for d1 in disease_list:
        for d2 in disease_list:
            if d1==d2: continue
            alpha = _co_occ_alpha(group_df, d1, d2)
            if alpha <= 0: continue
            i1, i2 = disease_list.index(d1), disease_list.index(d2)
            w = float(np.sum(np.abs(IF_adj[i1,:] * IF_adj[i2,:])) * alpha)
            if w > 0: G.add_edge(d1, d2, weight=w)
    return G

def _restrict_graph(G, nodes):
    H = nx.DiGraph(); H.add_nodes_from(nodes)
    for u,v,attr in G.edges(data=True):
        if (u in nodes) and (v in nodes): H.add_edge(u,v,**attr)
    return H

def _apply_chron(H, row):
    cf = {n:_chron_factor_from_row(row, n) for n in H.nodes()}
    Hc = nx.DiGraph()
    for n,val in cf.items(): Hc.add_node(n, chron=val)
    for u,v,attr in H.edges(data=True):
        base = attr.get('weight',0.0); edge_cf = np.sqrt(cf.get(u,0.0)*cf.get(v,0.0))
        Hc.add_edge(u,v, weight=base*edge_cf, base=base, cf=edge_cf)
    return Hc

def _node_strengths(G):
    s = {n:0.0 for n in G.nodes()}
    for u,v,attr in G.edges(data=True):
        w = attr.get('weight',0.0); s[u]+=w; s[v]+=w
    return s

def _scale(vals, a, b):
    vs = np.array(list(vals.values())) if isinstance(vals, dict) else np.array(vals)
    if vs.size==0 or np.max(vs)<=0:
        return {k:a for k in (vals.keys() if isinstance(vals, dict) else range(len(vals)))}
    vmax = np.max(vs)
    def f(x): return a + (b-a)*(x/(vmax+1e-12))
    if isinstance(vals, dict): return {k:f(v) for k,v in vals.items()}
    return [f(v) for v in vals]

def _edge_widths(G, lo=0.3, hi=5.0):
    ws = [G[u][v]['weight'] for u,v in G.edges()] if G.number_of_edges()>0 else []
    if not ws or max(ws)<=0: return []
    mx = max(ws)
    return [lo + (hi-lo)*(w/mx) for w in ws]

def _label_top_edges(ax, G, pos, top_k, which="ref"):
    if G.number_of_edges()==0: return
    edges_sorted = sorted(G.edges(data=True), key=lambda x: x[2].get('weight',0.0), reverse=True)[:top_k]
    labels = {}
    for u,v,attr in edges_sorted:
        if which=="both" and "base" in attr:
            labels[(u,v)] = f"ref={attr['base']:.3f}\nchron={attr['weight']:.3f}"
        else:
            labels[(u,v)] = f"{attr.get('weight',0.0):.3f}"
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=8, ax=ax, label_pos=0.5)

def _predict_logits_probs(head, X_np, node2v, gidx):
    X = torch.tensor(X_np, dtype=torch.float32, device=device)
    N = X.shape[0]; node2vN = node2v.expand(N,-1,-1); gidxN = gidx.expand(N)
    with torch.no_grad():
        a,v,e,h,_ = model(X, node2vN, gidxN)
        logit = {'asct':a,'vart':v,'encet':e,'hept':h}[head]
        p1 = torch.sigmoid(logit)
    return logit.detach().cpu().numpy().reshape(-1), p1.detach().cpu().numpy().reshape(-1)

# ===== Select patient (positional index within test split)
np.random.seed(0 if not RANDOM_PERSON else None)
if RANDOM_PERSON:
    PERSON_POS = np.random.randint(0, len(X_test_df))
pid_label = test_rows_df.index[PERSON_POS]
row_person = test_rows_df.iloc[PERSON_POS]

# Build reference group graph (IF × group multiplier)
gname = group_name_age_sex_other(row_person)
gidx = group_index_age_sex_other(row_person)
group_df = split_groups_age_sex_other(train_rows_df)[gname]
IF_adj = _get_IF_adj_for_group(gidx)
G_ref = build_group_graph_with_IFadj(group_df, IF_adj)            # ALL edges
present = [d for d in disease_list if (d in row_person.index) and (row_person[d]==1)]
H_ref = _restrict_graph(G_ref, present)
if H_ref.number_of_edges()==0:
    for u in present:
        for v in present:
            if u==v: continue
            iu,iv = disease_list.index(u), disease_list.index(v)
            base_w = float(np.sum(np.abs(IF_adj[iu,:]*IF_adj[iv,:])))
            if base_w>0: H_ref.add_edge(u,v,weight=base_w)
H_chron = _apply_chron(H_ref, row_person)

# ===== Predictions & correctness for this person
x_row = X_test_df.iloc[PERSON_POS].values.astype(np.float32).reshape(1,-1)
node2v_person = test_node2vec_emb_t[PERSON_POS:PERSON_POS+1]
gidx_person   = test_gidx_t[PERSON_POS:PERSON_POS+1]
y_true_vec = y_test_main_df.iloc[PERSON_POS][['ASCT','VART','ENCET','HEPT']].astype(int).values

pred_info = {}
for head in HEADS:
    lg, pr = _predict_logits_probs(head, x_row, node2v_person, gidx_person)
    T = THRESHOLDS[HEAD_TITLES[head]]
    yhat = int(pr[0] > T); ytrue = int(y_true_vec[['asct','vart','encet','hept'].index(head)])
    pred_info[head] = {'p':float(pr[0]), 'thr':T, 'yhat':yhat, 'ytrue':ytrue, 'ok': (yhat==ytrue)}

# ======== FIGURE 1: Reference group vs Personalized chronocity
pos = nx.spring_layout(G_ref, seed=46, k=0.9)
ns_ref = _scale(_node_strengths(G_ref), 500, 2600)
ns_pat = _scale(_node_strengths(H_ref), 700, 2200)
cf_nodes = {n:_chron_factor_from_row(row_person, n) for n in H_ref.nodes()}
ns_pat_ch = {n: ns_pat.get(n,900)*(0.6 + 0.8*cf_nodes.get(n,0.0)) for n in H_ref.nodes()}

fig1 = plt.figure(figsize=(18,6))
gs = fig1.add_gridspec(1,2, width_ratios=[2.2,1.0])
ax0 = fig1.add_subplot(gs[0,0]); ax1 = fig1.add_subplot(gs[0,1])

# ---- Group graph (NO edge-weight labels) ----
nx.draw_networkx_nodes(G_ref, pos,
                       node_size=[ns_ref[n] for n in G_ref.nodes()],
                       node_color='lightgray', edgecolors='k', linewidths=0.5, ax=ax0)
nx.draw_networkx_labels(G_ref, pos, font_size=9, ax=ax0)
w0 = _edge_widths(G_ref, lo=0.3, hi=4.5)
if w0:
    ws = [G_ref[u][v]['weight'] for u,v in G_ref.edges()]; mx = max(ws)
    alphas = [0.12 + 0.88*(w/mx) for w in ws]
    for (u,v), lw, a in zip(G_ref.edges(), w0, alphas):
        nx.draw_networkx_edges(G_ref, pos, edgelist=[(u,v)], width=lw, alpha=a,
                               arrows=True, arrowstyle='-|>', arrowsize=12, ax=ax0, edge_color='gray')
# IMPORTANT FIX (2): do NOT label group edges
# _label_top_edges(ax0, G_ref, pos, top_k=LABEL_TOP_GROUP, which="ref")
ax0.set_title(f"Reference group CIS — {gname}\n(IF × group-multiplier; all edges)"); ax0.axis('off')

# ---- Patient graph (keep ref vs chron labels) ----
node_cf_vals = [cf_nodes[n] for n in H_chron.nodes()]
nx.draw_networkx_nodes(H_chron, pos,
                       node_size=[ns_pat_ch[n] for n in H_chron.nodes()],
                       node_color=node_cf_vals, cmap='viridis', vmin=0, vmax=1,
                       edgecolors='k', linewidths=0.5, ax=ax1)
nx.draw_networkx_labels(H_chron, pos, font_size=9, ax=ax1)
w1 = _edge_widths(H_chron, lo=0.8, hi=5.0)
if w1:
    nx.draw_networkx_edges(H_chron, pos, width=w1, arrows=True, arrowstyle='-|>', arrowsize=12, ax=ax1)
_label_top_edges(ax1, H_chron, pos, top_k=LABEL_TOP_PERSON, which="both")
ax1.set_title("Patient CIS — chronocity weighted\n(edge label: ref vs chron)"); ax1.axis('off')
cbar = plt.colorbar(plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0,1)), ax=ax1,
                    fraction=0.046, pad=0.04); cbar.set_label('Chronocity factor')

# correctness box (kept as-is; currently empty)
lines = []
for h in HEADS:
    pi = pred_info[h]; tick = "✓" if pi['ok'] else "✗"
    # lines.append(f"{HEAD_TITLES[h]}: pred={pi['yhat']} true={pi['ytrue']} {tick}")
ax1.text(1.02, 0.5, "\n".join(lines), transform=ax1.transAxes, va='center',
         fontsize=10, bbox=dict(boxstyle="round,pad=0.4", fc="#f5f5f5", ec="#888"))

fig1.suptitle(f"Patient (test) pos={PERSON_POS}, id={pid_label} — present: {', '.join(present)}", y=0.98)
if SAVE_PNG:
    fig1.savefig(os.path.join(OUT_DIR, f"patient_{PERSON_POS}_{pid_label}_cis_ref_vs_chron.png"),
                 dpi=PNG_DPI, bbox_inches="tight")
# plt.tight_layout()
# plt.show()

# ======== FIGURE 2: Individual SHAP bars per target
feat_names_raw = list(X_train_df.columns)
feat_names_disp = [_pretty_feat_name(c) for c in feat_names_raw]   # IMPORTANT FIX (1)

bg_n = min(BG_SIZE, len(X_train_df))
bg_idx = np.random.choice(len(X_train_df), size=bg_n, replace=False)
bg_X = X_train_df.iloc[bg_idx].values.astype(np.float32)

def _make_shap_predict_fn(head):
    def _predict(X_np):
        lg, _ = _predict_logits_probs(head, X_np, node2v_person, gidx_person)
        return lg.reshape(-1,1)
    return _predict

fig2, axes = plt.subplots(2,2, figsize=(14,10), constrained_layout=True)
axes = axes.ravel()
for k, head in enumerate(HEADS):
    pi = pred_info[head]
    shap_expl = shap.KernelExplainer(_make_shap_predict_fn(head), bg_X)
    sv = shap_expl.shap_values(x_row, nsamples=NSAMPLES_SHAP)
    sv_vec = (sv[0] if isinstance(sv, list) else sv).reshape(-1)
    order = np.argsort(-np.abs(sv_vec))[:TOPK_SHAP]

    # Display names (Duration instead of Date)
    names = [feat_names_disp[i] for i in order][::-1]
    vals  = [sv_vec[i] for i in order][::-1]

    ax = axes[k]
    ax.barh(names, vals)
    ax.axvline(0.0, color='k', linewidth=1)
    ax.set_xlabel("SHAP on logit (signed)")
plt.suptitle("Individual SHAP explanations (per target) — reflects person’s actual graph/group", y=1.02)
if SAVE_PNG:
    fig2.savefig(os.path.join(OUT_DIR, f"patient_{PERSON_POS}_{pid_label}_shap_bars.png"),
                 dpi=PNG_DPI, bbox_inches="tight")
plt.show()

# ----------------------------------------------------------------------------
# 20B) Per-disease removal — annotated heatmap (for the same patient)
# ----------------------------------------------------------------------------
def _predict_probs_all(X_np, node2v, gidx):
    X = torch.tensor(X_np, dtype=torch.float32, device=device)
    N = X.shape[0]; node2vN = node2v.expand(N, -1, -1); gidxN = gidx.expand(N)
    with torch.no_grad():
        a, v, e, h, _ = model(X, node2vN, gidxN)
        P = torch.sigmoid(torch.stack([a,v,e,h], dim=1)).cpu().numpy().reshape(-1,4)
    return P[0]

pos = int(PERSON_POS)
x_base = X_test_df.iloc[pos].values.astype(np.float32).reshape(1,-1)
node2v_base = test_node2vec_emb_t[pos:pos+1]
gidx_base   = test_gidx_t[pos:pos+1]
row = test_rows_df.iloc[pos]
present = [d for d in disease_list if d in row.index and row[d]==1]
base_probs = _predict_probs_all(x_base, node2v_base, gidx_base)
TARGETS = ['ASCT','VART','ENCET','HEPT']

deltas = []
for d in present:
    x_cf = x_base.copy()
    if d in X_train_df.columns:
        j = list(X_train_df.columns).index(d); x_cf[0, j] = 0.0
    date_col = CHRON_DATE_MAP.get(d, None)
    if date_col and (date_col in X_train_df.columns):
        j = list(X_train_df.columns).index(date_col); x_cf[0, j] = 0.0
    node2v_cf = node2v_base.clone()
    if d in disease_list:
        d_pos = disease_list.index(d); node2v_cf[0, d_pos, :] = 0.0
    cf_probs = _predict_probs_all(x_cf, node2v_cf, gidx_base)
    deltas.append([d] + list(cf_probs - base_probs))

deltas_df = pd.DataFrame(deltas, columns=['Removed disease'] + TARGETS)
order = np.argsort(-np.abs(deltas_df['HEPT'].values))
deltas_df = deltas_df.iloc[order].reset_index(drop=True)

vals = deltas_df[TARGETS].values
vmax = np.max(np.abs(vals)) + 1e-12

fig3, ax = plt.subplots(
    figsize=(6.0, max(2.0, 0.40 * len(deltas_df))),
    constrained_layout=True
)

im = ax.imshow(vals, aspect='auto', cmap='bwr', vmin=-vmax, vmax=vmax)
ax.set_yticks(range(len(deltas_df))); ax.set_yticklabels(deltas_df['Removed disease'])
ax.set_xticks(range(len(TARGETS)));  ax.set_xticklabels(TARGETS)
ax.set_title(f'Per-disease removal — Δ predicted risk (patient pos {pos}, id {pid_label})')

cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Δ probability (cf − base)')

for i in range(vals.shape[0]):
    for j in range(vals.shape[1]):
        v = vals[i, j]
        ax.text(j, i, f"{v:+.3f}", ha='center', va='center',
                color=('white' if abs(v) > 0.5*vmax else 'black'), fontsize=9)

if SAVE_PNG:
    fig3.savefig(
        os.path.join(OUT_DIR, f"patient_{PERSON_POS}_{pid_label}_per_disease_removal.png"),
        dpi=PNG_DPI
    )
plt.show()

# =================== PT-ASSIST — per-patient decision-assist map ===================
import os, numpy as np, matplotlib.pyplot as plt, networkx as nx
from sklearn.linear_model import LogisticRegression
from matplotlib.patches import FancyArrowPatch

OUT_DIR = "target_relationships_figs"; os.makedirs(OUT_DIR, exist_ok=True)
ALL_TARGETS = ['ASCT','VART','ENCET','HEPT','DEADT']
MAIN_TARGETS = ['ASCT','VART','ENCET','HEPT']
THRESHOLDS = THRESHOLDS if 'THRESHOLDS' in globals() else {'ASCT':0.5,'VART':0.5,'ENCET':0.5,'HEPT':0.85}
THR_DEADT = 0.50

# ---------- 1) Get TRAIN & TEST probabilities for all 5 heads ----------
def _predict_all5(df_X, node2v, gidx, batch=128):
    N = len(df_X)
    probs = np.zeros((N,5), dtype=np.float32)
    X_np = df_X.values.astype(np.float32)
    for s in range(0, N, batch):
        e = min(N, s+batch)
        Xb = torch.tensor(X_np[s:e], dtype=torch.float32, device=device)
        n2v = node2v[s:e]; g = gidx[s:e]
        with torch.no_grad():
            a,v,e4,h,d = model(Xb, n2v, g)
            P = torch.sigmoid(torch.stack([a,v,e4,h,d], dim=1))
        probs[s:e] = P.detach().cpu().numpy()
    return probs

if 'probs_train_all5' not in globals():
    probs_train_all5 = _predict_all5(X_train_df, train_node2vec_emb_t, train_gidx_t)
if 'probs_test_all5' not in globals():
    probs_test_all5  = _predict_all5(X_test_df,  test_node2vec_emb_t,  test_gidx_t)

# labels
Y_train5 = np.zeros((len(X_train_df),5), dtype=int)
Y_train5[:, :4] = y_train_main_df[MAIN_TARGETS].astype(int).values
Y_train5[:, 4]  = y_train_aux_df['DEADT'].astype(int).values
Y_test5 = np.zeros((len(X_test_df),5), dtype=int)
Y_test5[:, :4] = y_test_main_df[MAIN_TARGETS].astype(int).values
Y_test5[:, 4]  = y_test_aux_df['DEADT'].astype(int).values

# ---------- 2) Fit stackers on TRAIN: y_B ~ probs(A≠B) ----------
stackers = {}
train_medians = np.median(probs_train_all5, axis=0)
for j, dst in enumerate(ALL_TARGETS):
    X = np.delete(probs_train_all5, j, axis=1)
    y = Y_train5[:, j]
    lr = LogisticRegression(max_iter=500, solver='lbfgs')
    lr.fit(X, y)
    stackers[dst] = lr

# ---------- 3) Pick patient ----------
PATIENT_ID = 23627 # example
if 'test_rows_df' in globals() and PATIENT_ID in test_rows_df.index:
    PERSON_POS = int(np.where(test_rows_df.index.values == PATIENT_ID)[0][0])
elif 'PERSON_POS' not in globals():
    PERSON_POS = np.random.randint(0, len(X_test_df))
pid_label = test_rows_df.index[PERSON_POS] if 'test_rows_df' in globals() else PERSON_POS

p_vec  = probs_test_all5[PERSON_POS, :]
y_vec  = Y_test5[PERSON_POS, :]
thrvec = np.array([THRESHOLDS.get(t, THR_DEADT) for t in ALL_TARGETS])
pred   = (p_vec > thrvec).astype(int)
correct= (pred == y_vec).astype(int)

# ---------- 4) For each edge A→B compute Δp(B) ----------
edges = []
DELTA_MIN = 0.03  # 3 percentage points
for j, dst in enumerate(ALL_TARGETS):
    X_full = np.delete(p_vec, j)  # (4,)
    base_p = stackers[dst].predict_proba(X_full.reshape(1,-1))[:,1][0]
    base_dec = int(base_p >= thrvec[j])

    for i, src in enumerate(ALL_TARGETS):
        if i == j: continue
        X_cf = X_full.copy()
        pos_in_X = i if i < j else i-1
        X_cf[pos_in_X] = train_medians[i]
        cf_p = stackers[dst].predict_proba(X_cf.reshape(1,-1))[:,1][0]
        delta = cf_p - base_p
        flip = (int(cf_p >= thrvec[j]) != base_dec)
        if flip or abs(delta) >= DELTA_MIN:
            edges.append((ALL_TARGETS[i], ALL_TARGETS[j], float(delta), bool(flip)))


# ---------- 5) Plot per-patient assist map  ----------
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D

def _scale_arr(arr, lo, hi):
    arr = np.asarray(arr, dtype=float)
    vmax = float(np.max(arr)) if np.max(arr) > 0 else 1.0
    return lo + (hi - lo) * (arr / (vmax + 1e-12))

def _face(p):
    return plt.cm.Greens(0.25 + 0.75 * float(p))

def _curvature(y1, y2):
    # small deterministic curvature so overlapping arrows separate a bit
    dy = y2 - y1
    if abs(dy) < 0.03:
        return 0.0
    return 0.10 * np.sign(dy) * min(1.8, 0.7 + abs(dy) / 0.12)

# -----------------------------
# controls for readability
# -----------------------------
SHOW_ONLY_FLIPS = True          # recommended for paper figure
MAX_CONTEXT_NONFLIP = 2         # only used if SHOW_ONLY_FLIPS = False
FIG_W, FIG_H = 12, 8

# node sizes
sizes = _scale_arr(p_vec, 900, 2200)

# order nodes to reduce clutter:
# positives first, then by predicted probability
idx_of = {t: i for i, t in enumerate(ALL_TARGETS)}
order = sorted(
    ALL_TARGETS,
    key=lambda t: (-int(pred[idx_of[t]]), -float(p_vec[idx_of[t]]), t)
)

# vertical positions
ys = np.linspace(0.88, 0.12, len(order))

# two-column layout
x_left = 0.18
x_right = 0.72
x_text_right = 0.80

pos_left = {t: (x_left, y) for t, y in zip(order, ys)}
pos_right = {t: (x_right, y) for t, y in zip(order, ys)}

# choose which edges to draw
flip_edges = [e for e in edges if e[3]]
nonflip_edges = [e for e in edges if not e[3]]

if SHOW_ONLY_FLIPS:
    edges_to_draw = sorted(flip_edges, key=lambda x: -abs(x[2]))
else:
    strongest_nonflip = sorted(nonflip_edges, key=lambda x: -abs(x[2]))[:MAX_CONTEXT_NONFLIP]
    edges_to_draw = sorted(flip_edges, key=lambda x: -abs(x[2])) + strongest_nonflip

# figure
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), constrained_layout=True)
ax.set_xlim(0.03, 1.02)
ax.set_ylim(0.03, 0.97)
ax.axis("off")

# column headers
ax.text(x_left, 0.94, "Source task perturbed", ha="center", va="bottom",
        fontsize=12, fontweight="bold")
ax.text(x_right, 0.94, "Affected prediction", ha="center", va="bottom",
        fontsize=12, fontweight="bold")

# subtle guide lines
ax.plot([x_left, x_left], [0.08, 0.92], color="lightgray", lw=1.0, zorder=0)
ax.plot([x_right, x_right], [0.08, 0.92], color="lightgray", lw=1.0, zorder=0)

# draw nodes
for t in order:
    k = idx_of[t]
    yl = pos_left[t][1]
    yr = pos_right[t][1]

    # left/source node (simple)
    ax.scatter([x_left], [yl],
               s=sizes[k] * 0.55,
               c=[_face(p_vec[k])],
               edgecolors="black",
               linewidths=1.2,
               zorder=3)
    ax.text(x_left - 0.035, yl, t,
            ha="right", va="center",
            fontsize=11, fontweight="bold", zorder=4)

    # right/affected node (full state)
    if y_vec[k] == 1:
        ax.scatter([x_right], [yr],
                   s=sizes[k] * 1.35,
                   c="gold", alpha=0.28, linewidths=0, zorder=1)

    ax.scatter([x_right], [yr],
               s=sizes[k],
               c=[_face(p_vec[k])],
               edgecolors=("#2ca02c" if correct[k] == 1 else "#d62728"),
               linewidths=2.2,
               zorder=4)

    # external label box instead of putting all text inside node
    label = (
        f"{t}\n"
        f"p={float(p_vec[k]):.2f}   pred={int(pred[k])}   true={int(y_vec[k])}   "
        f"{'✓' if correct[k] else '✗'}"
    )
    ax.text(x_text_right, yr, label,
            ha="left", va="center", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="0.82", alpha=0.97),
            zorder=5)

# draw edges
if len(edges_to_draw) == 0:
    ax.text(0.5, 0.03,
            "No decision-changing influences for this patient.",
            ha="center", va="bottom", fontsize=10)
else:
    m = max(abs(e[2]) for e in edges_to_draw) + 1e-12

    for (u, v, dp, flip) in edges_to_draw:
        y1 = pos_left[u][1]
        y2 = pos_right[v][1]

        color = "#d95f02" if dp > 0 else "#1f78b4"
        lw = 1.5 + 7.0 * (abs(dp) / m)
        alpha = 0.92 if flip else 0.20
        linestyle = "-" if flip else "--"
        rad = _curvature(y1, y2)

        patch = FancyArrowPatch(
            (x_left + 0.03, y1),
            (x_right - 0.03, y2),
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=lw,
            color=color,
            alpha=alpha,
            linestyle=linestyle,
            connectionstyle=f"arc3,rad={rad}",
            zorder=2
        )
        ax.add_patch(patch)

        # label flip edges only
        if flip:
            xm = 0.50
            ym = (y1 + y2) / 2.0

            # nudge label off the line a bit
            ym += 0.025 if y2 >= y1 else -0.025

            edge_label = f"{u} → {v}\nΔp={dp*100:+.1f} pp"
            ax.text(xm, ym, edge_label,
                    color=color, fontsize=9.5,
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=color, lw=0.8, alpha=0.96),
                    zorder=6)

# legend
legend_handles = [
    Line2D([0], [0], color="#d95f02", lw=2.5, label="Δp > 0"),
    Line2D([0], [0], color="#1f78b4", lw=2.5, label="Δp < 0"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
           markeredgecolor="#2ca02c", markeredgewidth=2, markersize=10,
           label="Correct prediction"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
           markeredgecolor="#d62728", markeredgewidth=2, markersize=10,
           label="Incorrect prediction"),
    Line2D([0], [0], marker="o", color="gold", alpha=0.35, lw=0, markersize=11,
           label="True label = 1 halo"),
]
ax.legend(handles=legend_handles, loc="lower center",
          bbox_to_anchor=(0.5, -0.01), ncol=5, frameon=False, fontsize=9)

# caption-like note
note = (
    "Shown: decision-changing edges only."
    if SHOW_ONLY_FLIPS
    else f"Shown: all flip edges + top {MAX_CONTEXT_NONFLIP} strongest non-flip edges."
)
ax.text(0.5, 0.01, note, ha="center", va="bottom", fontsize=10)

fname = os.path.join(OUT_DIR, f"PT_ASSIST_patient_{pid_label}_decision_map_clean.png")
fig.savefig(fname, dpi=240, bbox_inches="tight", pad_inches=0.10)
plt.show()
print("[PT-ASSIST] Saved:", fname)
