import os, json, time, random, glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


def unsorted_segment_sum(data, segment_ids, num_segments):
    result = data.new_full((num_segments, data.size(1)), 0)
    idx = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, idx, data)
    return result

def unsorted_segment_mean(data, segment_ids, num_segments):
    result = data.new_full((num_segments, data.size(1)), 0)
    count  = data.new_full((num_segments, data.size(1)), 0)
    idx    = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, idx, data)
    count.scatter_add_(0, idx, torch.ones_like(data))
    return result / count.clamp(min=1)


class E_GCL(nn.Module):
    def __init__(self, input_nf, output_nf, hidden_nf,
                 edges_in_d=0, act_fn=nn.SiLU(),
                 residual=True, attention=False,
                 normalize=False, coords_agg='mean', tanh=False):
        super().__init__()
        self.residual   = residual
        self.attention  = attention
        self.normalize  = normalize
        self.coords_agg = coords_agg
        self.epsilon    = 1e-8

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_nf * 2 + 1 + edges_in_d, hidden_nf), act_fn,
            nn.Linear(hidden_nf, hidden_nf), act_fn,
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf), act_fn,
            nn.Linear(hidden_nf, output_nf),
        )
        coord_layers = [nn.Linear(hidden_nf, hidden_nf), act_fn]
        last = nn.Linear(hidden_nf, 1, bias=False)
        nn.init.xavier_uniform_(last.weight, gain=0.001)
        coord_layers.append(last)
        if tanh:
            coord_layers.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_layers)
        if attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_nf, 1), nn.Sigmoid())

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        diff   = coord[row] - coord[col]
        radial = torch.sum(diff ** 2, dim=1, keepdim=True)
        if self.normalize:
            diff = diff / (torch.sqrt(radial).detach() + self.epsilon)
        return radial, diff

    def edge_model(self, source, target, radial, edge_attr):
        parts = [source, target, radial] + ([edge_attr] if edge_attr is not None else [])
        out = self.edge_mlp(torch.cat(parts, dim=1))
        if self.attention:
            out = out * self.att_mlp(out)
        return out

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, _ = edge_index
        trans  = coord_diff * self.coord_mlp(edge_feat)
        agg_fn = unsorted_segment_sum if self.coords_agg == 'sum' else unsorted_segment_mean
        return coord + agg_fn(trans, row, coord.size(0))

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, _ = edge_index
        agg  = unsorted_segment_sum(edge_attr, row, x.size(0))
        parts = [x, agg] + ([node_attr] if node_attr is not None else [])
        out  = self.node_mlp(torch.cat(parts, dim=1))
        return (x + out) if self.residual else out

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
        row, col           = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)
        edge_feat          = self.edge_model(h[row], h[col], radial, edge_attr)
        coord              = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        h                  = self.node_model(h, edge_index, edge_feat, node_attr)
        return h, coord, edge_attr


class EGNN_QM9(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf,
                 device='cpu', act_fn=nn.SiLU(), n_layers=7, attention=True):
        super().__init__()
        self.n_layers  = n_layers
        self.embedding = nn.Linear(in_node_nf, hidden_nf)
        for i in range(n_layers):
            self.add_module(f"gcl_{i}", E_GCL(
                hidden_nf, hidden_nf, hidden_nf,
                edges_in_d=in_edge_nf, act_fn=act_fn,
                residual=True, attention=attention,
            ))
        self.node_dec = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf), act_fn,
            nn.Linear(hidden_nf, hidden_nf),
        )
        self.graph_dec = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf), act_fn,
            nn.Linear(hidden_nf, 1),
        )
        self.to(device)

    def forward(self, h0, x, edges, edge_attr=None, node_mask=None, n_nodes=None):
        h = self.embedding(h0)
        for i in range(self.n_layers):
            h, x, _ = self._modules[f"gcl_{i}"](h, edges, x, edge_attr=edge_attr)
        h = self.node_dec(h)
        if node_mask is not None:
            h = h * node_mask
        B = h.size(0) // n_nodes
        h = h.view(B, n_nodes, -1)
        if node_mask is not None:
            mask = node_mask.view(B, n_nodes, 1)
            h = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        else:
            h = h.mean(1)
        return self.graph_dec(h).squeeze(-1)


_adj_cache = {}

def get_adj_matrix(n_nodes, batch_size, device):
    key = (n_nodes, batch_size, str(device))
    if key not in _adj_cache:
        rows, cols = zip(*[(i, j) for i in range(n_nodes)
                                   for j in range(n_nodes) if i != j])
        r = torch.LongTensor(rows)
        c = torch.LongTensor(cols)
        all_r = torch.cat([r + n_nodes * i for i in range(batch_size)])
        all_c = torch.cat([c + n_nodes * i for i in range(batch_size)])
        _adj_cache[key] = [all_r.to(device), all_c.to(device)]
    return _adj_cache[key]


ATOM_ENCODER  = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
NUM_ATOM_TYPES = len(ATOM_ENCODER)
SYM_TO_Z      = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}
QM9_PROPS     = ['mu','alpha','homo','lumo','gap','r2','zpve','U0','U','H','G','Cv']

_KNOWN_XYZ_DIR = '/kaggle/input/quantum-machine-9-aka-qm9/dsgdb9nsd.xyz'


def _find_xyz_dir():
    if (os.path.isdir(_KNOWN_XYZ_DIR) and
            len(glob.glob(os.path.join(_KNOWN_XYZ_DIR, 'dsgdb9nsd_*.xyz'))) > 100):
        return _KNOWN_XYZ_DIR

    for dirpath, _, filenames in os.walk('/kaggle/input'):
        n = sum(1 for f in filenames
                if f.startswith('dsgdb9nsd_') and f.endswith('.xyz'))
        if n > 100:
            return dirpath

    raise FileNotFoundError(
        "\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "  QM9 xyz files not found. The pickle has no geometry.\n"
        "  You must add the zaharch dataset to this notebook:\n\n"
        "  1. Click 'Add data' in the top-right of your Kaggle notebook\n"
        "  2. Search: zaharch quantum machine 9\n"
        "  3. Add 'Quantum Machine 9, aka QM9' by zaharch\n"
        "  4. Re-run the notebook\n\n"
        "  After adding, files will be at:\n"
        f"  {_KNOWN_XYZ_DIR}/dsgdb9nsd_000001.xyz\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    )


def _parse_float(s):
    return float(s.replace('*^', 'e'))


def _parse_xyz_positions(path):
    try:
        with open(path) as f:
            lines = f.readlines()
        na   = int(lines[0].strip())
        syms, pos = [], []
        for i in range(na):
            toks = lines[2 + i].strip().split()
            syms.append(toks[0])
            pos.append([_parse_float(toks[1]),
                        _parse_float(toks[2]),
                        _parse_float(toks[3])])
        return syms, np.array(pos, dtype=np.float32)
    except Exception:
        return None, None


PICKLE_PATH = '/kaggle/input/notebooks/zaharch/quantum-machine-9-qm9/data.covs.pickle'


def load_dataset(pickle_path=PICKLE_PATH, max_samples=None, verbose=True):
    xyz_dir = _find_xyz_dir()
    n_xyz = len(glob.glob(os.path.join(xyz_dir, 'dsgdb9nsd_*.xyz')))
    if verbose:
        print(f"  QM9 xyz files: {xyz_dir}  ({n_xyz:,} files)")

    if not os.path.isfile(pickle_path):
        raise FileNotFoundError(f"Pickle not found: {pickle_path}")

    if verbose:
        print(f"  Reading pickle: {pickle_path}")
    df = pd.read_pickle(pickle_path)

    prop_cols = [c for c in QM9_PROPS if c in df.columns]
    mol_props = df.groupby('molecule_name')[prop_cols].first().reset_index()
    if verbose:
        print(f"  {len(df):,} coupling rows → {len(mol_props):,} unique molecules")
        print(f"  Properties available: {prop_cols}")

    if max_samples:
        mol_props = mol_props.iloc[:max_samples]

    if verbose:
        print(f"  Parsing xyz geometry …")

    data_list, n_skipped = [], 0

    for _, row in mol_props.iterrows():
        name  = row['molecule_name']
        fpath = os.path.join(xyz_dir, name + '.xyz')

        if not os.path.isfile(fpath):
            n_skipped += 1
            continue

        syms, positions = _parse_xyz_positions(fpath)
        if syms is None:
            n_skipped += 1
            continue

        if not set(syms).issubset(ATOM_ENCODER):
            n_skipped += 1
            continue

        charges = np.array([float(SYM_TO_Z[s]) for s in syms], dtype=np.float32)

        mol = {
            'n_atoms'  : len(syms),
            'positions': positions,
            'charges'  : charges,
        }
        for p in QM9_PROPS:
            mol[p] = float(row[p]) if p in prop_cols else 0.0

        data_list.append(mol)

        if verbose and len(data_list) % 10000 == 0:
            print(f"    … {len(data_list):,} molecules loaded")

    if verbose:
        print(f"  Done. {len(data_list):,} molecules "
              f"(skipped {n_skipped} missing/unsupported)\n")
    return data_list


def _synthetic_fallback(n=500):
    data_list = []
    for _ in range(n):
        na = np.random.randint(3, 10)
        mol = {'n_atoms': na,
               'positions': np.random.randn(na, 3).astype(np.float32),
               'charges'  : np.random.choice([1,6,7,8], size=na).astype(np.float32)}
        for p in QM9_PROPS:
            mol[p] = float(np.random.randn())
        data_list.append(mol)
    return data_list


class QM9Dataset(Dataset):
    def __init__(self, data_list, property='homo', max_atoms=29):
        self.data     = data_list
        self.property = property
        self.N        = max_atoms

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        mol = self.data[idx]
        n, N = mol['n_atoms'], self.N

        pos      = np.zeros((N, 3), dtype=np.float32)
        one_hot  = np.zeros((N, NUM_ATOM_TYPES), dtype=np.float32)
        charges  = np.zeros((N, 1), dtype=np.float32)
        atom_mask= np.zeros((N, 1), dtype=np.float32)

        pos[:n]        = mol['positions'][:n]
        charges[:n, 0] = mol['charges'][:n]
        atom_mask[:n, 0] = 1.0

        z_to_sym = {1:'H', 6:'C', 7:'N', 8:'O', 9:'F'}
        for i, z in enumerate(mol['charges'][:n]):
            sym = z_to_sym.get(int(z), 'C')
            if sym in ATOM_ENCODER:
                one_hot[i, ATOM_ENCODER[sym]] = 1.0

        return {
            'positions' : torch.FloatTensor(pos),
            'one_hot'   : torch.FloatTensor(one_hot),
            'charges'   : torch.FloatTensor(charges),
            'atom_mask' : torch.FloatTensor(atom_mask),
            self.property: torch.FloatTensor([mol[self.property]]),
        }


def split_dataset(data_list, train_size=110000, val_size=10000,
                  test_size=10000, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(data_list))
    return ([data_list[i] for i in idx[:train_size]],
            [data_list[i] for i in idx[train_size:train_size+val_size]],
            [data_list[i] for i in idx[train_size+val_size:train_size+val_size+test_size]])


def preprocess_input(one_hot, charges, charge_power, charge_scale, device):
    if one_hot.dim() == 3:
        one_hot = one_hot.view(-1, one_hot.size(-1))
    if charges.dim() == 3:
        charges = charges.view(-1, 1)
    charge_tensor = (charges / charge_scale).pow(
        torch.arange(1, charge_power + 1, device=device, dtype=torch.float32))
    return torch.cat([one_hot, charge_tensor], dim=1)


def compute_mean_mad(loader, prop_name):
    values = torch.cat([b[prop_name] for b in loader]).float()
    mean   = values.mean().item()
    mad    = (values - mean).abs().mean().item()
    print(f"  Property '{prop_name}':  mean={mean:.6f}  MAD={mad:.6f}")
    return mean, mad


def _forward(model, batch, n_nodes, charge_power, charge_scale, device):
    B   = batch['positions'].size(0)
    pos = batch['positions'].view(B * n_nodes, -1).to(device)
    mask= batch['atom_mask'].view(B * n_nodes, -1).to(device)
    nodes = preprocess_input(
        batch['one_hot'].to(device),
        batch['charges'].to(device),
        charge_power, charge_scale, device
    ).view(B * n_nodes, -1)
    edges = get_adj_matrix(n_nodes, B, device)
    return model(h0=nodes, x=pos, edges=edges, node_mask=mask, n_nodes=n_nodes)


def train_epoch(model, loader, optimizer, loss_fn, mean, mad,
                prop, charge_power, charge_scale, n_nodes, device):
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        optimizer.zero_grad()
        pred  = _forward(model, batch, n_nodes, charge_power, charge_scale, device)
        label = batch[prop].to(device).squeeze(-1)
        loss  = loss_fn(pred, (label - mean) / mad)
        loss.backward(); optimizer.step()
        total += loss.item() * batch['positions'].size(0)
        n     += batch['positions'].size(0)
    return total / n


@torch.no_grad()
def eval_epoch(model, loader, loss_fn, mean, mad,
               prop, charge_power, charge_scale, n_nodes, device):
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        pred  = _forward(model, batch, n_nodes, charge_power, charge_scale, device)
        label = batch[prop].to(device).squeeze(-1)
        total += loss_fn(mad * pred + mean, label).item() * batch['positions'].size(0)
        n     += batch['positions'].size(0)
    return total / n


@torch.no_grad()
def eval_test_detailed(model, loader, mean, mad,
                       prop, charge_power, charge_scale, n_nodes, device):
    model.eval()
    preds, labels = [], []
    for batch in loader:
        pred  = _forward(model, batch, n_nodes, charge_power, charge_scale, device)
        label = batch[prop].to(device).squeeze(-1)
        preds.append((mad * pred + mean).cpu())
        labels.append(label.cpu())

    preds  = torch.cat(preds)
    labels = torch.cat(labels)
    abs_err = (preds - labels).abs()
    rel_err = abs_err / labels.abs().clamp(min=1e-8) * 100.0
    n = len(labels)
    return dict(
        mae          = abs_err.mean().item(),
        rmse         = (preds - labels).pow(2).mean().sqrt().item(),
        mape         = rel_err.mean().item(),
        within_1pct  = (rel_err <  1.0).sum().item() / n * 100,
        within_5pct  = (rel_err <  5.0).sum().item() / n * 100,
        within_10pct = (rel_err < 10.0).sum().item() / n * 100,
        n_total      = n,
    )


def main():
    PROPERTY     = 'homo'
    LR           = 1e-3
    BATCH_SIZE   = 96
    EPOCHS       = 1000
    N_LAYERS     = 7
    HIDDEN_NF    = 128
    ATTENTION    = True
    CHARGE_POWER = 2
    MAX_ATOMS    = 29
    NUM_WORKERS  = 2
    WEIGHT_DECAY = 1e-16

    if os.path.isfile(PICKLE_PATH):
        all_data = load_dataset(verbose=True)
    else:
        print(f"⚠  Pickle not found at:\n   {PICKLE_PATH}\n"
              "   Running on synthetic data.\n")
        all_data = _synthetic_fallback(500)

    n_total = len(all_data)
    if n_total >= 130000:
        _tr, _va, _te = 110000, 10000, 10000
    else:
        _tr = max(BATCH_SIZE, int(n_total * 0.8))
        _va = max(BATCH_SIZE, int(n_total * 0.1))
        _te = max(BATCH_SIZE, n_total - _tr - _va)

    train_data, val_data, test_data = split_dataset(
        all_data, train_size=_tr, val_size=_va, test_size=_te)
    print(f"Split → train={len(train_data):,}  val={len(val_data):,}  test={len(test_data):,}\n")

    mk = lambda d, s: DataLoader(
        QM9Dataset(d, property=PROPERTY, max_atoms=MAX_ATOMS),
        batch_size=BATCH_SIZE, shuffle=s, num_workers=NUM_WORKERS,
        drop_last=(s and len(d) > BATCH_SIZE))
    loaders = dict(train=mk(train_data, True),
                   valid=mk(val_data,   False),
                   test =mk(test_data,  False))

    charge_scale = max(float(max(d['charges'].max() for d in train_data)), 9.0)

    print("Dataset statistics:")
    mean, mad = compute_mean_mad(loaders['train'], PROPERTY)
    print()

    in_node_nf = NUM_ATOM_TYPES + CHARGE_POWER
    model = EGNN_QM9(in_node_nf=in_node_nf, in_edge_nf=0, hidden_nf=HIDDEN_NF,
                     device=DEVICE, act_fn=nn.SiLU(),
                     n_layers=N_LAYERS, attention=ATTENTION)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: EGNN_QM9  ({n_params:,} trainable parameters)")
    print(f"Target: {PROPERTY}  |  Epochs: {EPOCHS}  |  LR: {LR}\n")

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)
    loss_l1   = nn.L1Loss()

    os.makedirs('/kaggle/working/egnn_logs', exist_ok=True)
    best_val, best_epoch, history = float('inf'), 0, []

    print(f"{'Epoch':>6}  {'Train MAE':>10}  {'Val MAE':>10}  {'Best Val':>10}  {'Time':>6}")
    print("─" * 55)

    for epoch in range(1, EPOCHS + 1):
        t0        = time.time()
        train_mae = train_epoch(model, loaders['train'], optimizer, loss_l1,
                                mean, mad, PROPERTY, CHARGE_POWER,
                                charge_scale, MAX_ATOMS, DEVICE)
        scheduler.step()
        val_mae   = eval_epoch(model, loaders['valid'], loss_l1,
                               mean, mad, PROPERTY, CHARGE_POWER,
                               charge_scale, MAX_ATOMS, DEVICE)

        history.append({'epoch': epoch, 'train_mae': train_mae, 'val_mae': val_mae})

        if val_mae < best_val:
            best_val, best_epoch = val_mae, epoch
            torch.save(model.state_dict(), '/kaggle/working/egnn_best.pt')

        print(f"{epoch:>6}  {train_mae:>10.5f}  {val_mae:>10.5f}  "
              f"{best_val:>10.5f}  {time.time()-t0:>5.1f}s")

    with open('/kaggle/working/egnn_logs/history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n{'═'*55}")
    print(f"Training done. Loading best checkpoint (epoch {best_epoch}) …")
    model.load_state_dict(torch.load('/kaggle/working/egnn_best.pt', map_location=DEVICE))

    res = eval_test_detailed(model, loaders['test'], mean, mad,
                             PROPERTY, CHARGE_POWER, charge_scale, MAX_ATOMS, DEVICE)
    n = res['n_total']
    print(f"\n{'═'*55}")
    print(f"  TEST RESULTS  —  '{PROPERTY}'  ({n:,} molecules)")
    print(f"{'═'*55}")
    print(f"  MAE                          : {res['mae']:.6f}")
    print(f"  RMSE                         : {res['rmse']:.6f}")
    print(f"  MAPE (mean abs % error)      : {res['mape']:.2f} %")
    print(f"{'─'*55}")
    print(f"  Molecules within  1% error   : {res['within_1pct']:.1f} %")
    print(f"  Molecules within  5% error   : {res['within_5pct']:.1f} %")
    print(f"  Molecules within 10% error   : {res['within_10pct']:.1f} %")
    print(f"{'═'*55}")
    print(f"  Best validation MAE          : {best_val:.6f}  (epoch {best_epoch})")
    print(f"{'═'*55}\n")

    summary = dict(**res, best_val_mae=best_val,
                   best_val_epoch=best_epoch, property=PROPERTY)
    with open('/kaggle/working/egnn_logs/test_results.json', 'w') as f:
        json.dump(summary, f, indent=2)


def smoke_test():
    print("── Smoke test: EGNN_QM9 ────────────────────────────────────────────")
    model = EGNN_QM9(in_node_nf=7, in_edge_nf=0, hidden_nf=64,
                     device='cpu', n_layers=2, attention=True)
    N, B = 10, 2
    pred = model(h0=torch.randn(B*N, 7), x=torch.randn(B*N, 3),
                 edges=get_adj_matrix(N, B, 'cpu'),
                 node_mask=torch.ones(B*N, 1), n_nodes=N)
    assert pred.shape == (B,)
    print(f"  pred shape: {pred.shape}  ✓")
    print("  All checks passed ✓\n")


if __name__ == "__main__":
    smoke_test()
    main()
else:
    smoke_test()