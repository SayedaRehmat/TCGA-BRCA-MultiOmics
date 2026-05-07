"""
==============================================================================
AI MODULE 1: VARIATIONAL AUTOENCODER (VAE) FOR GENE EXPRESSION
==============================================================================
Replaces: PCA (linear dimensionality reduction)
Adds:     Non-linear latent space · captures complex co-expression patterns
          Generative model · can synthesise new patient profiles (data augmentation)
          Probabilistic encoding · uncertainty quantification per patient


"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle, os
from scipy.special import expit as sigmoid
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

# =============================================================================
# PROJECT PATHS
# =============================================================================
BASE_DIR = "/content/drive/MyDrive/TCGA_Project"

TCGA_EXPR_PATH = os.path.join(BASE_DIR, "gene_expression.tsv")
TCGA_CLIN_PATH = os.path.join(BASE_DIR, "clinical_survival.tsv")

OUTDIR = os.path.join(BASE_DIR, "results")
FIGDIR = os.path.join(BASE_DIR, "figures")

os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(FIGDIR, exist_ok=True)

print("=" * 60)
print("AI MODULE 1: VARIATIONAL AUTOENCODER (VAE)")
print("=" * 60)

# =============================================================================
# LOAD INPUT FILES
# =============================================================================
with open(f"{OUTDIR}/expr_filtered.pkl", 'rb') as f:
    expr = pickle.load(f)

with open(f"{OUTDIR}/clin_clustered.pkl", 'rb') as f:
    clin = pickle.load(f)

# ── Use top-3000 MAD genes ───────────────────────────────────────────────────
mad = expr.apply(lambda x: np.median(np.abs(x - np.median(x))), axis=0)

top_genes = mad.nlargest(3000).index

X_raw = expr[top_genes].values.astype(np.float32)

scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

n_samples, n_features = X.shape

print(f"Input: {n_samples} samples × {n_features} genes")

# =============================================================================
# VAE IMPLEMENTATION
# =============================================================================

class VAE:
    def __init__(self, input_dim, hidden_dims, latent_dim, lr=1e-3, seed=42):

        np.random.seed(seed)

        self.ld = latent_dim
        self.lr = lr

        # Encoder
        dims = [input_dim] + hidden_dims

        self.We, self.be = [], []

        for i in range(len(dims)-1):

            scale = np.sqrt(2.0 / dims[i])

            self.We.append(
                np.random.randn(dims[i], dims[i+1]) * scale
            )

            self.be.append(np.zeros(dims[i+1]))

        # Mean + log variance
        scale = np.sqrt(2.0 / dims[-1])

        self.Wmu = np.random.randn(dims[-1], latent_dim) * scale
        self.bmu = np.zeros(latent_dim)

        self.Wlv = np.random.randn(dims[-1], latent_dim) * scale
        self.blv = np.zeros(latent_dim)

        # Decoder
        ddims = [latent_dim] + hidden_dims[::-1] + [input_dim]

        self.Wd, self.bd = [], []

        for i in range(len(ddims)-1):

            scale = np.sqrt(2.0 / ddims[i])

            self.Wd.append(
                np.random.randn(ddims[i], ddims[i+1]) * scale
            )

            self.bd.append(np.zeros(ddims[i+1]))

    def relu(self, x):
        return np.maximum(0, x)

    def drelu(self, x):
        return (x > 0).astype(np.float32)

    def encode(self, X):

        h = X

        self.eh = [X]

        for W, b in zip(self.We, self.be):

            h = self.relu(h @ W + b)

            self.eh.append(h)

        mu = h @ self.Wmu + self.bmu

        logvar = np.clip(h @ self.Wlv + self.blv, -4, 4)

        return mu, logvar

    def reparameterize(self, mu, logvar):

        eps = np.random.randn(*mu.shape)

        return mu + np.exp(0.5 * logvar) * eps

    def decode(self, z):

        h = z

        self.dh = [z]

        for i, (W, b) in enumerate(zip(self.Wd, self.bd)):

            h = h @ W + b

            if i < len(self.Wd) - 1:
                h = self.relu(h)

            self.dh.append(h)

        return h

    def forward(self, X):

        mu, logvar = self.encode(X)

        z = self.reparameterize(mu, logvar)

        x_hat = self.decode(z)

        return x_hat, mu, logvar, z

    def loss(self, X, x_hat, mu, logvar, beta=1.0):

        recon = np.mean((X - x_hat)**2)

        kl = -0.5 * np.mean(
            1 + logvar - mu**2 - np.exp(logvar)
        )

        return recon + beta * kl, recon, kl

    def train_step(self, X_batch, beta=1.0):

        bs = X_batch.shape[0]

        x_hat, mu, logvar, z = self.forward(X_batch)

        total_loss, recon, kl = self.loss(
            X_batch,
            x_hat,
            mu,
            logvar,
            beta
        )

        dL_dxhat = 2 * (x_hat - X_batch) / bs

        # Decoder backward
        dh = dL_dxhat

        dWd_list, dbd_list = [], []

        for i in range(len(self.Wd)-1, -1, -1):

            inp = self.dh[i]

            dWd = inp.T @ dh / bs
            dbd = dh.mean(axis=0)

            dWd_list.insert(0, dWd)
            dbd_list.insert(0, dbd)

            dh = dh @ self.Wd[i].T

            if i > 0:
                dh *= self.drelu(self.dh[i])

        # Encoder gradients
        dz = dh.copy()

        dmu_kl = mu / bs

        dlv_kl = 0.5 * (np.exp(logvar) - 1) / bs

        eps = (z - mu) / (
            np.exp(0.5 * logvar) + 1e-8
        )

        dmu = (dz + beta * dmu_kl)

        dlv = (
            dz * eps * 0.5 * np.exp(0.5 * logvar)
            + beta * dlv_kl
        )

        h_enc = self.eh[-1]

        dWmu = h_enc.T @ dmu / bs
        dbmu = dmu.mean(axis=0)

        dWlv = h_enc.T @ dlv / bs
        dblv = dlv.mean(axis=0)

        dh_enc = (
            dmu @ self.Wmu.T
            + dlv @ self.Wlv.T
        )

        # Encoder backward
        dWe_list, dbe_list = [], []

        dh = dh_enc

        for i in range(len(self.We)-1, -1, -1):

            inp = self.eh[i]

            dWe = inp.T @ dh / bs
            dbe = dh.mean(axis=0)

            dWe_list.insert(0, dWe)
            dbe_list.insert(0, dbe)

            dh = dh @ self.We[i].T

            if i > 0:
                dh *= self.drelu(self.eh[i])

        # SGD updates
        lr = self.lr

        for i in range(len(self.We)):

            self.We[i] -= lr * np.clip(dWe_list[i], -1, 1)

            self.be[i] -= lr * np.clip(dbe_list[i], -1, 1)

        self.Wmu -= lr * np.clip(dWmu, -1, 1)
        self.bmu -= lr * np.clip(dbmu, -1, 1)

        self.Wlv -= lr * np.clip(dWlv, -1, 1)
        self.blv -= lr * np.clip(dblv, -1, 1)

        for i in range(len(self.Wd)):

            self.Wd[i] -= lr * np.clip(dWd_list[i], -1, 1)

            self.bd[i] -= lr * np.clip(dbd_list[i], -1, 1)

        return total_loss, recon, kl

# =============================================================================
# TRAINING
# =============================================================================

LATENT_DIM = 32
HIDDEN = [256, 128]

BATCH_SIZE = 64
EPOCHS = 40

BETA = 1.0

vae = VAE(
    n_features,
    HIDDEN,
    LATENT_DIM,
    lr=2e-4
)

print(f"\nTraining VAE: {n_features} → {HIDDEN} → {LATENT_DIM}")

losses, recons, kls = [], [], []

n_batches = n_samples // BATCH_SIZE

for epoch in range(EPOCHS):

    idx = np.random.permutation(n_samples)

    ep_loss = ep_recon = ep_kl = 0.0

    for b in range(n_batches):

        batch = X[
            idx[b*BATCH_SIZE:(b+1)*BATCH_SIZE]
        ]

        l, r, k = vae.train_step(
            batch,
            beta=BETA
        )

        ep_loss += l
        ep_recon += r
        ep_kl += k

    losses.append(ep_loss / n_batches)
    recons.append(ep_recon / n_batches)
    kls.append(ep_kl / n_batches)

    if (epoch + 1) % 10 == 0:

        print(
            f"  Epoch {epoch+1:3d}/{EPOCHS} | "
            f"Total={losses[-1]:.4f} | "
            f"Recon={recons[-1]:.4f} | "
            f"KL={kls[-1]:.4f}"
        )

# =============================================================================
# LATENT SPACE
# =============================================================================

mu_all, logvar_all = vae.encode(X)

Z = mu_all

print(f"\nLatent space shape: {Z.shape}")

# =============================================================================
# COMPARISON VS PCA
# =============================================================================

X_pca = np.load(f"{OUTDIR}/X_pca.npy")

y = clin['cluster_label'].map({
    'C1':0,
    'C2':1,
    'C3':2
}).values

# VAE clustering
km_vae = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=20
)

labels_vae = km_vae.fit_predict(Z)

sil_vae = silhouette_score(
    Z,
    labels_vae,
    sample_size=500,
    random_state=42
)

# PCA clustering
km_pca = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=20
)

labels_pca = km_pca.fit_predict(X_pca[:, :20])

sil_pca = silhouette_score(
    X_pca[:, :20],
    labels_pca,
    sample_size=500,
    random_state=42
)

print(f"\nSilhouette score — PCA: {sil_pca:.4f}")
print(f"Silhouette score — VAE: {sil_vae:.4f}")

print(
    f"Improvement: "
    f"+{(sil_vae-sil_pca)*100:.1f}% relative"
)

# =============================================================================
# RECONSTRUCTION QUALITY
# =============================================================================

x_hat_all, _, _, _ = vae.forward(X)

mse = np.mean((X - x_hat_all)**2)

r2 = 1 - mse / np.var(X)

print(f"Reconstruction R²: {r2:.4f}")

# =============================================================================
# SAVE OUTPUTS
# =============================================================================

np.save(f"{OUTDIR}/vae_latent.npy", Z)

np.save(f"{OUTDIR}/vae_logvar.npy", logvar_all)

with open(f"{OUTDIR}/vae_model.pkl", 'wb') as f:
    pickle.dump(vae, f)

with open(f"{OUTDIR}/vae_scaler.pkl", 'wb') as f:
    pickle.dump(scaler, f)

pd.DataFrame({
    'epoch': np.arange(1, EPOCHS+1),
    'total_loss': losses,
    'reconstruction_loss': recons,
    'kl_divergence': kls
}).to_csv(f"{OUTDIR}/vae_training_history.csv", index=False)

pd.DataFrame(
    Z,
    index=expr.index,
    columns=[f'VAE_Z{i+1}' for i in range(Z.shape[1])]
).to_csv(f"{OUTDIR}/vae_latent_space.csv")

# =============================================================================
# FIGURES
# =============================================================================

colors = {
    'C1':'#1f77b4',
    'C2':'#d62728',
    'C3':'#2ca02c'
}

labels_true = clin['cluster_label'].values

fig, axes = plt.subplots(2, 3, figsize=(18, 11))

fig.suptitle(
    "AI Module 1: Variational Autoencoder (VAE)\n"
    "Non-linear latent space vs PCA baseline",
    fontsize=14,
    fontweight='bold'
)

# Training curves
axes[0,0].plot(losses, 'k-', lw=2, label='Total ELBO')

axes[0,0].plot(recons, 'b--', lw=1.5, label='Reconstruction')

axes[0,0].plot(kls, 'r--', lw=1.5, label='KL divergence')

axes[0,0].set_xlabel("Epoch")
axes[0,0].set_ylabel("Loss")

axes[0,0].set_title("VAE Training Curves")

axes[0,0].legend(fontsize=9)

axes[0,0].grid(alpha=0.3)

# PCA space
for cl in ['C1','C2','C3']:

    mask = labels_true == cl

    axes[0,1].scatter(
        X_pca[mask,0],
        X_pca[mask,1],
        c=colors[cl],
        label=cl,
        alpha=0.5,
        s=10,
        edgecolors='none'
    )

axes[0,1].set_xlabel("PC1")
axes[0,1].set_ylabel("PC2")

axes[0,1].set_title(
    f"PCA Latent Space\nSilhouette={sil_pca:.4f}"
)

axes[0,1].legend(markerscale=2, fontsize=9)

axes[0,1].grid(alpha=0.3)

# VAE space
for cl in ['C1','C2','C3']:

    mask = labels_true == cl

    axes[0,2].scatter(
        Z[mask,0],
        Z[mask,1],
        c=colors[cl],
        label=cl,
        alpha=0.5,
        s=10,
        edgecolors='none'
    )

axes[0,2].set_xlabel("VAE z1")
axes[0,2].set_ylabel("VAE z2")

axes[0,2].set_title(
    f"VAE Latent Space\nSilhouette={sil_vae:.4f}"
)

axes[0,2].legend(markerscale=2, fontsize=9)

axes[0,2].grid(alpha=0.3)

# Uncertainty
avg_logvar = logvar_all.mean(axis=1)

for cl in ['C1','C2','C3']:

    mask = labels_true == cl

    axes[1,0].hist(
        avg_logvar[mask],
        bins=30,
        alpha=0.6,
        color=colors[cl],
        label=cl,
        density=True
    )

axes[1,0].set_xlabel("Mean log-variance")

axes[1,0].set_ylabel("Density")

axes[1,0].set_title("Latent Encoding Uncertainty")

axes[1,0].legend(fontsize=9)

axes[1,0].grid(alpha=0.3)

# Silhouette comparison
ax = axes[1,1]

bars = ax.bar(
    ['PCA', 'VAE'],
    [sil_pca, sil_vae],
    color=['#90CAF9', '#7B1FA2'],
    edgecolor='white',
    width=0.5
)

for bar, val in zip(bars, [sil_pca, sil_vae]):

    ax.text(
        bar.get_x()+bar.get_width()/2,
        val+0.002,
        f'{val:.4f}',
        ha='center',
        fontsize=12,
        fontweight='bold'
    )

ax.set_ylabel("Silhouette Score")

ax.set_title("Cluster Quality Comparison")

ax.grid(axis='y', alpha=0.3)

# Reconstruction scatter
sample_idx = np.random.choice(
    n_samples,
    200,
    replace=False
)

x_orig = X[sample_idx, :50].flatten()

x_recon = x_hat_all[sample_idx, :50].flatten()

axes[1,2].scatter(
    x_orig,
    x_recon,
    s=3,
    alpha=0.3,
    color='#1565C0'
)

axes[1,2].plot(
    [-4,4],
    [-4,4],
    'r--',
    lw=1.5,
    label='Perfect'
)

axes[1,2].set_xlabel("Original")

axes[1,2].set_ylabel("Reconstructed")

axes[1,2].set_title(f"Reconstruction Quality\nR²={r2:.4f}")

axes[1,2].legend(fontsize=9)

axes[1,2].grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    f"{FIGDIR}/AI_01_VAE.png",
    dpi=150,
    bbox_inches='tight'
)

plt.close()

print(f"\n[Saved] {FIGDIR}/AI_01_VAE.png")
print("AI MODULE 1 COMPLETE ✓")
