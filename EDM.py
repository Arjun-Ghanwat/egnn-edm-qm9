import os, glob, time, json, math, random, copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


def unsorted_segment_sum(data, segment_ids, num_segments):
    result = data.new_zeros(num_segments, data.size(1))
    idx    = segment_ids.unsqueeze(-1).expand_as(data)
    result.scatter_add_(0, idx, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    result = data.new_zeros(num_segments, data.size(1))
    count  = data.new_zeros(num_segments, data.size(1))
    idx    = segment_ids.unsqueeze(-1).expand_as(data)
    result.scatter_add_(0, idx, data)
    count.scatter_add_(0, idx, torch.ones_like(data))
    return result / count.clamp(min=1)


class GCL(nn.Module):

    def __init__(self, input_nf, output_nf, hidden_nf,
                 edges_in_d=2, act_fn=nn.SiLU(), attention=True,
                 aggregation_method='sum', normalization_factor=1.0):
        super().__init__()
        self.aggregation_method   = aggregation_method
        self.normalization_factor = normalization_factor
        self.attention            = attention

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_nf * 2 + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf),
        )
        if attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_nf, 1), nn.Sigmoid())

    def edge_model(self, h_i, h_j, edge_attr, edge_mask):
        out = self.edge_mlp(torch.cat([h_i, h_j, edge_attr], dim=1))
        if self.attention:
            out = out * self.att_mlp(out)
        if edge_mask is not None:
            out = out * edge_mask
        return out

    def node_model(self, h, edge_index, m_ij):
        row = edge_index[0]
        if self.aggregation_method == 'sum':
            agg = unsorted_segment_sum(m_ij, row, h.size(0))
        else:
            agg = unsorted_segment_mean(m_ij, row, h.size(0))
        agg = agg / self.normalization_factor
        return self.node_mlp(torch.cat([h, agg], dim=1))

    def forward(self, h, edge_index, edge_attr, edge_mask=None):
        row, col = edge_index
        m_ij     = self.edge_model(h[row], h[col], edge_attr, edge_mask)
        h_out    = self.node_model(h, edge_index, m_ij)
        return h_out, m_ij


class EquivariantCoordUpdate(nn.Module):

    def __init__(self, hidden_nf, act_fn=nn.SiLU(), tanh=True,
                 coords_range=15.0, aggregation_method='sum',
                 normalization_factor=1.0):
        super().__init__()
        self.tanh                 = tanh
        self.coords_range         = coords_range
        self.aggregation_method   = aggregation_method
        self.normalization_factor = normalization_factor

        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf), act_fn,
            nn.Linear(hidden_nf, 1, bias=False),
        )
        if tanh:
            self.coord_mlp = nn.Sequential(
                nn.Linear(hidden_nf, hidden_nf), act_fn,
                nn.Linear(hidden_nf, 1, bias=False),
                nn.Tanh(),
            )
        nn.init.xavier_uniform_(self.coord_mlp[-2 if tanh else -1].weight, gain=0.001)

    def forward(self, x, edge_index, coord_diff, m_ij, edge_mask=None):
        row = edge_index[0]
        weights = self.coord_mlp(m_ij)
        if self.tanh:
            weights = weights * self.coords_range
        trans = coord_diff * weights
        if edge_mask is not None:
            trans = trans * edge_mask
        if self.aggregation_method == 'sum':
            agg = unsorted_segment_sum(trans, row, x.size(0))
        else:
            agg = unsorted_segment_mean(trans, row, x.size(0))
        return x + agg / self.normalization_factor


class EGNN_Layer(nn.Module):

    def __init__(self, hidden_nf, edges_in_d=2, act_fn=nn.SiLU(),
                 attention=True, norm_diff=True, tanh=True,
                 coords_range=15.0, aggregation_method='sum',
                 normalization_factor=1.0):
        super().__init__()
        self.norm_diff = norm_diff

        self.gcl = GCL(
            hidden_nf, hidden_nf, hidden_nf,
            edges_in_d=edges_in_d, act_fn=act_fn,
            attention=attention, aggregation_method=aggregation_method,
            normalization_factor=normalization_factor,
        )
        self.coord_update = EquivariantCoordUpdate(
            hidden_nf, act_fn=act_fn, tanh=tanh,
            coords_range=coords_range,
            aggregation_method=aggregation_method,
            normalization_factor=normalization_factor,
        )

    def forward(self, h, x, edge_index, node_mask=None, edge_mask=None):
        row, col   = edge_index
        coord_diff = x[row] - x[col]
        radial     = (coord_diff ** 2).sum(1, keepdim=True)

        if self.norm_diff:
            norm       = (radial.sqrt() + 1e-8).detach()
            coord_diff = coord_diff / norm

        edge_attr = torch.cat([radial / (radial + 1.0), radial], dim=1)
        if edge_mask is not None:
            edge_attr = edge_attr * edge_mask

        h, m_ij = self.gcl(h, edge_index, edge_attr, edge_mask)
        x       = self.coord_update(x, edge_index, coord_diff, m_ij, edge_mask)

        if node_mask is not None:
            h = h * node_mask
        return h, x


class EGNN_Dynamics(nn.Module):

    def __init__(self, in_node_nf, time_emb_dim=32, hidden_nf=256,
                 n_layers=9, act_fn=nn.SiLU(), attention=True,
                 norm_diff=True, tanh=True, coords_range=15.0,
                 aggregation_method='sum', normalization_factor=1.0,
                 n_dims=3):
        super().__init__()
        self.n_dims      = n_dims
        self.n_layers    = n_layers
        self.in_node_nf  = in_node_nf
        self.time_emb_dim= time_emb_dim

        self.node_embedding = nn.Linear(in_node_nf + time_emb_dim, hidden_nf)
        self.node_output    = nn.Linear(hidden_nf, in_node_nf)

        for i in range(n_layers):
            self.add_module(f"layer_{i}", EGNN_Layer(
                hidden_nf, edges_in_d=2, act_fn=act_fn,
                attention=attention, norm_diff=norm_diff, tanh=tanh,
                coords_range=coords_range,
                aggregation_method=aggregation_method,
                normalization_factor=normalization_factor,
            ))

    @staticmethod
    def _sinusoidal_time_emb(t_int, T, dim, device):
        half = dim // 2
        freq = math.log(10000) / (half - 1)
        freq = torch.exp(torch.arange(half, device=device).float() * -freq)
        t_f  = t_int.float() / T
        emb  = t_f[:, None] * freq[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=1)

    def forward(self, h, x, t_int, T, edge_index, node_mask=None, edge_mask=None):
        B_N = h.size(0)
        B   = t_int.size(0)
        N   = B_N // B

        t_emb = self._sinusoidal_time_emb(t_int, T, self.time_emb_dim, h.device)
        t_emb = t_emb.repeat_interleave(N, dim=0)

        h_in = torch.cat([h, t_emb], dim=1)
        h    = self.node_embedding(h_in)
        if node_mask is not None:
            h = h * node_mask

        x_in = x.clone()

        for i in range(self.n_layers):
            h, x = self._modules[f"layer_{i}"](
                h, x, edge_index, node_mask=node_mask, edge_mask=edge_mask)

        x_eps = x - x_in
        x_eps = _remove_mean_with_mask_batched(x_eps, node_mask, B, N)

        h_eps = self.node_output(h)
        if node_mask is not None:
            h_eps = h_eps * node_mask

        return h_eps, x_eps


def _remove_mean_with_mask_batched(x, node_mask, B, N):
    x_r = x.view(B, N, 3)
    m_r = node_mask.view(B, N, 1) if node_mask is not None else torch.ones_like(x_r[:, :, :1])
    com = (x_r * m_r).sum(1, keepdim=True) / m_r.sum(1, keepdim=True).clamp(min=1)
    x_r = x_r - com
    if node_mask is not None:
        x_r = x_r * m_r
    return x_r.view(B * N, 3)


_adj_cache = {}

def get_adj_matrix(n_nodes, batch_size, device):
    key = (n_nodes, batch_size, str(device))
    if key not in _adj_cache:
        rows, cols = zip(*[(i, j) for i in range(n_nodes)
                                   for j in range(n_nodes) if i != j])
        r = torch.LongTensor(rows)
        c = torch.LongTensor(cols)
        all_r = torch.cat([r + n_nodes * k for k in range(batch_size)])
        all_c = torch.cat([c + n_nodes * k for k in range(batch_size)])
        _adj_cache[key] = [all_r.to(device), all_c.to(device)]
    return _adj_cache[key]


def make_edge_mask(node_mask, edge_index):
    row, col   = edge_index
    edge_mask  = node_mask[row] * node_mask[col]
    return edge_mask


def polynomial_schedule(timesteps, s=1e-5, power=2.0):
    T     = timesteps
    steps = T + 1
    t     = torch.linspace(0, T, steps)
    alpha2 = (1.0 - (t / T) ** power) ** 2
    alpha2 = alpha2.clamp(min=s)
    alpha2 = alpha2 / alpha2[0]
    return alpha2.sqrt()


def cosine_schedule(timesteps, s=0.008):
    T     = timesteps
    steps = T + 1
    t     = torch.linspace(0, T, steps)
    alpha_bar = torch.cos(((t / T) + s) / (1.0 + s) * math.pi * 0.5) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    return alpha_bar.sqrt().clamp(min=0.0, max=1.0)


class EDM(nn.Module):

    def __init__(self,
                 in_node_nf     = 6,
                 hidden_nf      = 256,
                 n_layers        = 9,
                 timesteps       = 1000,
                 noise_schedule  = 'polynomial_2',
                 noise_precision = 1e-5,
                 time_emb_dim    = 32,
                 attention       = True,
                 norm_values     = (1., 4., 10.),
                 n_dims          = 3,
                 device          = 'cpu'):
        super().__init__()

        self.in_node_nf    = in_node_nf
        self.n_dims        = n_dims
        self.T             = timesteps
        self.norm_values   = norm_values
        self.time_emb_dim  = time_emb_dim
        self.device_str    = str(device)

        self.dynamics = EGNN_Dynamics(
            in_node_nf           = in_node_nf,
            time_emb_dim         = time_emb_dim,
            hidden_nf            = hidden_nf,
            n_layers             = n_layers,
            act_fn               = nn.SiLU(),
            attention            = attention,
            norm_diff            = True,
            tanh                 = True,
            coords_range         = 15.0,
            aggregation_method   = 'sum',
            normalization_factor = 1.0,
            n_dims               = n_dims,
        )

        if noise_schedule == 'cosine':
            alpha_bar = cosine_schedule(timesteps, s=noise_precision)
        elif noise_schedule.startswith('polynomial'):
            power     = float(noise_schedule.split('_')[1]) if '_' in noise_schedule else 2.0
            alpha_bar = polynomial_schedule(timesteps, s=noise_precision, power=power)
        else:
            raise ValueError(f"Unknown noise schedule: {noise_schedule}")

        sigma_bar = (1.0 - alpha_bar ** 2).clamp(min=0.0).sqrt()

        self.register_buffer('alpha_bar', alpha_bar)
        self.register_buffer('sigma_bar', sigma_bar)

        alpha2 = alpha_bar ** 2
        sigma2 = sigma_bar ** 2
        beta_t = torch.zeros(timesteps + 1)
        for t in range(1, timesteps + 1):
            num = sigma2[t - 1]
            den = sigma2[t].clamp(min=1e-10)
            fac = (1.0 - alpha2[t] / alpha2[t - 1].clamp(min=1e-10)).clamp(min=0.0)
            beta_t[t] = (num / den * fac).clamp(max=0.999)
        self.register_buffer('beta_t', beta_t)

        self.to(device)

    def _norm_x(self, x):   return x / self.norm_values[0]
    def _norm_h(self, h):   return h / self.norm_values[1]
    def _unnorm_x(self, x): return x * self.norm_values[0]
    def _unnorm_h(self, h): return h * self.norm_values[1]

    def compute_loss(self, x0, h0, t_int, node_mask, edge_index, edge_mask, B, N):
        a   = self.alpha_bar[t_int]
        s   = self.sigma_bar[t_int]
        a_n = a.repeat_interleave(N).unsqueeze(1)
        s_n = s.repeat_interleave(N).unsqueeze(1)

        eps_x = torch.randn_like(x0)
        eps_h = torch.randn_like(h0)

        eps_x = _remove_mean_with_mask_batched(eps_x, node_mask, B, N)

        z_x = a_n * x0 + s_n * eps_x
        z_h = a_n * h0 + s_n * eps_h
        z_x = z_x * node_mask
        z_h = z_h * node_mask

        eps_h_pred, eps_x_pred = self.dynamics(
            h=z_h, x=z_x, t_int=t_int, T=self.T,
            edge_index=edge_index, node_mask=node_mask, edge_mask=edge_mask,
        )

        loss_x = ((eps_x - eps_x_pred) ** 2) * node_mask
        loss_h = ((eps_h - eps_h_pred) ** 2) * node_mask

        n_active = node_mask.sum().clamp(min=1)
        loss = (loss_x.sum() * self.n_dims + loss_h.sum() * self.in_node_nf) / \
               (n_active * (self.n_dims + self.in_node_nf))
        return loss

    @torch.no_grad()
    def sample(self, n_samples, n_nodes, device):
        B = n_samples
        N = n_nodes
        node_mask  = torch.ones(B * N, 1, device=device)
        edge_index = get_adj_matrix(N, B, device)
        edge_mask  = make_edge_mask(node_mask, edge_index)

        z_x = torch.randn(B * N, self.n_dims, device=device)
        z_h = torch.randn(B * N, self.in_node_nf, device=device)
        z_x = _remove_mean_with_mask_batched(z_x, node_mask, B, N)

        for t_val in reversed(range(1, self.T + 1)):
            t_int = torch.full((B,), t_val, device=device, dtype=torch.long)
            a  = self.alpha_bar[t_val]
            s  = self.sigma_bar[t_val]
            a0 = self.alpha_bar[t_val - 1]
            s0 = self.sigma_bar[t_val - 1]

            eps_h, eps_x = self.dynamics(
                h=z_h, x=z_x, t_int=t_int, T=self.T,
                edge_index=edge_index, node_mask=node_mask, edge_mask=edge_mask,
            )

            z0_x = (z_x - s * eps_x) / a.clamp(min=1e-8)
            z0_h = (z_h - s * eps_h) / a.clamp(min=1e-8)

            mu_x = a0 * (s ** 2) / (s ** 2 + (a * s0) ** 2 + 1e-10) * z0_x + \
                   a  * (s0 ** 2) / (s ** 2 + (a * s0) ** 2 + 1e-10) * z_x
            mu_h = a0 * (s ** 2) / (s ** 2 + (a * s0) ** 2 + 1e-10) * z0_h + \
                   a  * (s0 ** 2) / (s ** 2 + (a * s0) ** 2 + 1e-10) * z_h

            noise_scale = self.beta_t[t_val].sqrt()
            if t_val > 1:
                z_x = mu_x + noise_scale * torch.randn_like(mu_x)
                z_h = mu_h + noise_scale * torch.randn_like(mu_h)
            else:
                z_x, z_h = mu_x, mu_h

            z_x = _remove_mean_with_mask_batched(z_x, node_mask, B, N)

        z_x = self._unnorm_x(z_x)
        z_h = self._unnorm_h(z_h)
        return z_x.view(B, N, 3), z_h.view(B, N, self.in_node_nf)


class EMA:

    def __init__(self, model, decay=0.9999):
        self.model  = model
        self.decay  = decay
        self.shadow = {k: v.clone().float()
                       for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self):
        d = self.decay
        for k, v in self.model.state_dict().items():
            if self.shadow[k].dtype.is_floating_point:
                self.shadow[k] = d * self.shadow[k] + (1.0 - d) * v.float()

    def apply_shadow(self):
        self._backup = copy.deepcopy(self.model.state_dict())
        new_sd = {k: v.to(self.model.state_dict()[k].device).to(self.model.state_dict()[k].dtype)
                  for k, v in self.shadow.items()}
        self.model.load_state_dict(new_sd)

    def restore(self):
        self.model.load_state_dict(self._backup)


ATOM_ENCODER   = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
NUM_ATOM_TYPES = len(ATOM_ENCODER)
SYM_TO_Z       = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}
QM9_PROPS      = ['mu','alpha','homo','lumo','gap','r2','zpve','U0','U','H','G','Cv']

PICKLE_PATH    = '/kaggle/input/notebooks/zaharch/quantum-machine-9-qm9/data.covs.pickle'
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
        f"  Files should appear at: {_KNOWN_XYZ_DIR}/dsgdb9nsd_000001.xyz\n"
    )


def _parse_float(s):
    return float(s.replace('*^', 'e'))


def _parse_xyz(path):
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


def load_dataset(pickle_path=PICKLE_PATH, max_samples=None, verbose=True):
    xyz_dir = _find_xyz_dir()
    n_xyz   = len(glob.glob(os.path.join(xyz_dir, 'dsgdb9nsd_*.xyz')))
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

        syms, positions = _parse_xyz(fpath)
        if syms is None or not set(syms).issubset(ATOM_ENCODER):
            n_skipped += 1
            continue

        charges = np.array([float(SYM_TO_Z[s]) for s in syms], dtype=np.float32)
        mol = {'n_atoms': len(syms), 'positions': positions, 'charges': charges}
        for p in QM9_PROPS:
            mol[p] = float(row[p]) if p in prop_cols else 0.0
        data_list.append(mol)

        if verbose and len(data_list) % 10000 == 0:
            print(f"    … {len(data_list):,} molecules loaded")

    if verbose:
        print(f"  Done. {len(data_list):,} molecules "
              f"(skipped {n_skipped} missing/unsupported)\n")
    return data_list


def _synthetic_fallback(n=200):
    data_list = []
    for _ in range(n):
        na  = np.random.randint(3, 10)
        mol = {'n_atoms': na,
               'positions': np.random.randn(na, 3).astype(np.float32),
               'charges':   np.random.choice([1,6,7,8], size=na).astype(np.float32)}
        for p in QM9_PROPS:
            mol[p] = float(np.random.randn())
        data_list.append(mol)
    return data_list


def split_dataset(data_list, train_size=110000, val_size=10000,
                  test_size=10000, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(data_list))
    return (
        [data_list[i] for i in idx[:train_size]],
        [data_list[i] for i in idx[train_size:train_size + val_size]],
        [data_list[i] for i in idx[train_size + val_size:train_size + val_size + test_size]],
    )


class QM9DatasetEDM(Dataset):
    def __init__(self, data_list, max_atoms=29):
        self.data = data_list
        self.N    = max_atoms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        mol = self.data[idx]
        n, N = mol['n_atoms'], self.N

        pos       = np.zeros((N, 3), np.float32)
        one_hot   = np.zeros((N, NUM_ATOM_TYPES), np.float32)
        charges   = np.zeros((N, 1), np.float32)
        atom_mask = np.zeros((N, 1), np.float32)

        pos[:n]          = mol['positions'][:n]
        charges[:n, 0]   = mol['charges'][:n]
        atom_mask[:n, 0] = 1.0

        z_to_sym = {1:'H', 6:'C', 7:'N', 8:'O', 9:'F'}
        for i, z in enumerate(mol['charges'][:n]):
            sym = z_to_sym.get(int(z), 'C')
            if sym in ATOM_ENCODER:
                one_hot[i, ATOM_ENCODER[sym]] = 1.0

        return {
            'positions': torch.FloatTensor(pos),
            'one_hot'  : torch.FloatTensor(one_hot),
            'charges'  : torch.FloatTensor(charges),
            'atom_mask': torch.FloatTensor(atom_mask),
        }


def _forward_loss(model, batch, max_atoms, device):
    B   = batch['positions'].size(0)
    N   = max_atoms

    x         = batch['positions'].to(device).view(B * N, 3)
    one_hot   = batch['one_hot'].to(device).view(B * N, NUM_ATOM_TYPES)
    charges   = batch['charges'].to(device).view(B * N, 1)
    node_mask = batch['atom_mask'].to(device).view(B * N, 1)

    x = _remove_mean_with_mask_batched(x, node_mask, B, N)

    h = torch.cat([one_hot, charges / 9.0], dim=1)

    x_norm = x   / model.norm_values[0]
    h_norm = h   / model.norm_values[1]
    x_norm = x_norm * node_mask
    h_norm = h_norm * node_mask

    edge_index = get_adj_matrix(N, B, device)
    edge_mask  = make_edge_mask(node_mask, edge_index)

    t_int = torch.randint(1, model.T + 1, (B,), device=device)

    return model.compute_loss(
        x0=x_norm, h0=h_norm, t_int=t_int,
        node_mask=node_mask,
        edge_index=edge_index, edge_mask=edge_mask,
        B=B, N=N,
    )


def train_epoch(model, loader, optimizer, ema, max_atoms, device):
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        optimizer.zero_grad()
        loss = _forward_loss(model, batch, max_atoms, device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if ema is not None:
            ema.update()
        total += loss.item() * batch['positions'].size(0)
        n     += batch['positions'].size(0)
    return total / n


@torch.no_grad()
def eval_epoch(model, loader, max_atoms, device, ema=None):
    if ema is not None:
        ema.apply_shadow()
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        loss   = _forward_loss(model, batch, max_atoms, device)
        total += loss.item() * batch['positions'].size(0)
        n     += batch['positions'].size(0)
    if ema is not None:
        ema.restore()
    return total / n


def smoke_test():
    print("── Smoke test: EDM ─────────────────────────────────────────────────")
    model = EDM(
        in_node_nf=6, hidden_nf=64, n_layers=2,
        timesteps=10, noise_schedule='polynomial_2',
        attention=True, device='cpu',
    )

    B, N = 2, 10
    x         = torch.randn(B * N, 3)
    one_hot   = F.one_hot(torch.randint(0, 5, (B * N,)), 5).float()
    charges   = torch.randint(1, 9, (B * N, 1)).float()
    node_mask = torch.ones(B * N, 1)
    h         = torch.cat([one_hot, charges / 9.0], dim=1)

    edge_index = get_adj_matrix(N, B, 'cpu')
    edge_mask  = make_edge_mask(node_mask, edge_index)
    t_int      = torch.randint(1, 11, (B,))

    loss = model.compute_loss(
        x0=x / model.norm_values[0],
        h0=h / model.norm_values[1],
        t_int=t_int,
        node_mask=node_mask,
        edge_index=edge_index, edge_mask=edge_mask,
        B=B, N=N,
    )
    assert loss.shape == torch.Size([]), f"Expected scalar, got {loss.shape}"
    assert not torch.isnan(loss), "Loss is NaN!"
    print(f"  loss: {loss.item():.5f}  ✓")
    print("  All checks passed ✓\n")


def main():
    DIFFUSION_STEPS  = 1000
    NOISE_SCHEDULE   = 'polynomial_2'
    NOISE_PRECISION  = 1e-5
    HIDDEN_NF        = 256
    N_LAYERS         = 9
    ATTENTION        = True
    LR               = 1e-4
    BATCH_SIZE       = 128
    EPOCHS           = 3000
    EMA_DECAY        = 0.9999
    MAX_ATOMS        = 29
    NUM_WORKERS      = 2
    NORM_VALUES      = (1., 4., 10.)
    IN_NODE_NF       = NUM_ATOM_TYPES + 1
    TIME_EMB_DIM     = 32

    if os.path.isfile(PICKLE_PATH):
        all_data = load_dataset(verbose=True)
    else:
        print(f"⚠  Pickle not found at:\n   {PICKLE_PATH}\n"
              "   Running on synthetic data.\n")
        all_data = _synthetic_fallback(200)

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
        QM9DatasetEDM(d, max_atoms=MAX_ATOMS),
        batch_size=BATCH_SIZE, shuffle=s,
        num_workers=NUM_WORKERS, pin_memory=True,
        drop_last=(s and len(d) > BATCH_SIZE),
    )
    loaders = dict(
        train = mk(train_data, True),
        valid = mk(val_data,   False),
        test  = mk(test_data,  False),
    )

    model = EDM(
        in_node_nf     = IN_NODE_NF,
        hidden_nf      = HIDDEN_NF,
        n_layers       = N_LAYERS,
        timesteps      = DIFFUSION_STEPS,
        noise_schedule = NOISE_SCHEDULE,
        noise_precision= NOISE_PRECISION,
        time_emb_dim   = TIME_EMB_DIM,
        attention      = ATTENTION,
        norm_values    = NORM_VALUES,
        n_dims         = 3,
        device         = DEVICE,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: EDM  ({n_params:,} trainable parameters)")
    print(f"Diffusion steps: {DIFFUSION_STEPS}  |  Schedule: {NOISE_SCHEDULE}  |"
          f"  Epochs: {EPOCHS}  |  LR: {LR}\n")

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)
    ema       = EMA(model, decay=EMA_DECAY)

    os.makedirs('/kaggle/working/edm_logs', exist_ok=True)
    best_val, best_epoch, history = float('inf'), 0, []

    print(f"{'Epoch':>6}  {'Train Loss':>11}  {'Val Loss':>11}  {'Best Val':>11}  {'Time':>6}")
    print("─" * 58)

    for epoch in range(1, EPOCHS + 1):
        t0         = time.time()
        train_loss = train_epoch(model, loaders['train'], optimizer, ema,
                                 MAX_ATOMS, DEVICE)
        scheduler.step()
        val_loss   = eval_epoch(model, loaders['valid'], MAX_ATOMS, DEVICE, ema)

        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss})

        if val_loss < best_val:
            best_val, best_epoch = val_loss, epoch
            torch.save({
                'epoch'     : epoch,
                'model_state': model.state_dict(),
                'ema_shadow' : ema.shadow,
                'optimizer'  : optimizer.state_dict(),
            }, '/kaggle/working/edm_best.pt')

        elapsed = time.time() - t0
        print(f"{epoch:>6}  {train_loss:>11.5f}  {val_loss:>11.5f}  "
              f"{best_val:>11.5f}  {elapsed:>5.1f}s")

    with open('/kaggle/working/edm_logs/history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n{'═' * 58}")
    print(f"Training done. Loading best checkpoint (epoch {best_epoch}) …")
    ckpt = torch.load('/kaggle/working/edm_best.pt', map_location=DEVICE)
    model.load_state_dict(ckpt['model_state'])
    ema.shadow = ckpt['ema_shadow']

    test_loss = eval_epoch(model, loaders['test'], MAX_ATOMS, DEVICE, ema)

    print(f"\n{'═' * 58}")
    print(f"  TEST RESULTS  —  EDM  (QM9 generation)")
    print(f"{'═' * 58}")
    print(f"  Test diffusion loss (L2, EMA) : {test_loss:.6f}")
    print(f"{'─' * 58}")
    print(f"  Best validation loss          : {best_val:.6f}  (epoch {best_epoch})")
    print(f"{'═' * 58}\n")

    summary = dict(
        test_loss      = test_loss,
        best_val_loss  = best_val,
        best_epoch     = best_epoch,
        model          = 'EDM',
        diffusion_steps= DIFFUSION_STEPS,
        noise_schedule = NOISE_SCHEDULE,
        hidden_nf      = HIDDEN_NF,
        n_layers       = N_LAYERS,
        n_params       = n_params,
    )
    with open('/kaggle/working/edm_logs/test_results.json', 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    smoke_test()
    main()
else:
    smoke_test()