"""
Sinir agi tarafi: cok katmanli algilayici (MLP) + egitim dongusu.

Varsayilan backend PyTorch (CPU). torch kurulu degilse otomatik olarak
scikit-learn'un MLPClassifier'ina duser, boylece proje her durumda calisir.

Tasarim notlari
---------------
* Dengesiz siniflar var (CONTROL 138 vs ADHD 45) -> sinif agirlikli
  cross-entropy kullaniyoruz.
* Kucuk N (~250) -> guclu duzenlilestirme: dropout + weight decay +
  ic dogrulama kumesinde erken durdurma.
* Her fold icin ayri tohum (seed) veriyoruz; sonuclar tekrar uretilebilir.
"""
from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except Exception:                                  # pragma: no cover
    HAS_TORCH = False


# ==========================================================================
if HAS_TORCH:

    class MLP(nn.Module):
        """Girdi -> [Linear-BN-ReLU-Dropout] x L -> Linear(n_classes)."""

        def __init__(self, in_dim: int, n_classes: int,
                     hidden=(256, 64), dropout: float = 0.4):
            super().__init__()
            layers, prev = [], in_dim
            for h in hidden:
                layers += [nn.Linear(prev, h), nn.BatchNorm1d(h),
                           nn.ReLU(), nn.Dropout(dropout)]
                prev = h
            layers.append(nn.Linear(prev, n_classes))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)


def set_threads(n: int | None) -> None:
    if HAS_TORCH and n:
        torch.set_num_threads(int(n))


# ==========================================================================
def _stratified_val_split(y: np.ndarray, val_fraction: float, seed: int):
    """Her sinifin en az 1 ornegi dogrulamada olacak sekilde stratified bolme."""
    rng = np.random.default_rng(seed)
    val_idx = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        k = max(1, int(round(len(idx) * val_fraction)))
        k = min(k, len(idx) - 1) if len(idx) > 1 else 0
        val_idx.extend(idx[:k].tolist())
    val_idx = np.array(sorted(val_idx), dtype=int)
    mask = np.ones(len(y), dtype=bool)
    mask[val_idx] = False
    return np.where(mask)[0], val_idx


def train_mlp(X_train: np.ndarray, y_train: np.ndarray, n_classes: int,
              params: dict, seed: int = 0, record_history: bool = False):
    """
    MLP egit ve (model, history) dondur.

    X_train : (n, d) float32, ONCEDEN olceklenmis olmali (StandardScaler).
    y_train : (n,) int, 0..n_classes-1
    """
    if not HAS_TORCH:
        return _train_sklearn_mlp(X_train, y_train, params, seed)

    p = dict(params)
    torch.manual_seed(seed)
    np.random.seed(seed)

    tr_idx, va_idx = _stratified_val_split(y_train, p["val_fraction"], seed)
    Xtr = torch.tensor(X_train[tr_idx], dtype=torch.float32)
    ytr = torch.tensor(y_train[tr_idx], dtype=torch.long)
    Xva = torch.tensor(X_train[va_idx], dtype=torch.float32)
    yva = torch.tensor(y_train[va_idx], dtype=torch.long)

    counts = np.bincount(y_train[tr_idx], minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = torch.tensor(counts.sum() / (n_classes * counts), dtype=torch.float32)

    model = MLP(X_train.shape[1], n_classes,
                hidden=p["hidden"], dropout=p["dropout"])
    opt = torch.optim.Adam(model.parameters(), lr=p["lr"],
                           weight_decay=p["weight_decay"])
    crit = nn.CrossEntropyLoss(weight=weights)

    n = Xtr.shape[0]
    bs = min(int(p["batch_size"]), max(n, 2))
    best_loss, best_state, bad = np.inf, None, 0
    history = {"train_loss": [], "val_loss": []}
    g = torch.Generator().manual_seed(seed)

    for epoch in range(int(p["max_epochs"])):
        model.train()
        perm = torch.randperm(n, generator=g)
        epoch_loss, nb = 0.0, 0
        for s in range(0, n, bs):
            b = perm[s:s + bs]
            if len(b) < 2:                       # BatchNorm en az 2 ornek ister
                continue
            opt.zero_grad()
            loss = crit(model(Xtr[b]), ytr[b])
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())
            nb += 1

        model.eval()
        with torch.no_grad():
            vl = float(crit(model(Xva), yva).item()) if len(va_idx) else epoch_loss / max(nb, 1)
        if record_history:
            history["train_loss"].append(epoch_loss / max(nb, 1))
            history["val_loss"].append(vl)

        if vl < best_loss - 1e-5:
            best_loss, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= int(p["patience"]):
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, history


def predict_proba(model, X: np.ndarray) -> np.ndarray:
    """Egitilmis modelden sinif olasiliklari."""
    if not HAS_TORCH or not isinstance(model, MLP):
        return model.predict_proba(X)
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
        return torch.softmax(logits, dim=1).numpy()


# ==========================================================================
def _train_sklearn_mlp(X_train, y_train, params, seed):
    """torch yoksa yedek: sklearn MLPClassifier."""
    from sklearn.neural_network import MLPClassifier
    p = dict(params)
    clf = MLPClassifier(
        hidden_layer_sizes=tuple(p["hidden"]),
        alpha=p["weight_decay"],
        learning_rate_init=p["lr"],
        batch_size=min(int(p["batch_size"]), len(y_train)),
        max_iter=int(p["max_epochs"]),
        early_stopping=True,
        n_iter_no_change=int(p["patience"]),
        validation_fraction=p["val_fraction"],
        random_state=seed,
    )
    clf.fit(X_train, y_train)
    hist = {"train_loss": list(getattr(clf, "loss_curve_", [])),
            "val_loss": list(getattr(clf, "validation_scores_", []) or [])}
    return clf, hist


def backend_name() -> str:
    return f"PyTorch {torch.__version__}" if HAS_TORCH else "scikit-learn MLPClassifier"
