import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
from matplotlib.colors import LinearSegmentedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score
import statsmodels.api as sm
from scipy import stats
import warnings, os

warnings.filterwarnings('ignore')

# ── Paden ────────────────────────────────────────────────────────────────────
INPUT_PATH = r"C:\Users\Gebruiker\Documents\DATA\Resultaten\DATASET_zonderGSNB.csv"
OUTPUT_DIR = r"C:\Users\Gebruiker\Documents\DATA\Grafieken\Regressie_PCA"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLR_NEG    = "#C0392B"
CLR_POS    = "#2980B9"
CLR_ACCENT = "#2C3E50"
CLR_LIGHT  = "#ECF0F1"
CLR_PC1    = "#8E44AD"
CLR_PC2    = "#27AE60"

PRED_NAMES = ["max_G", "max_M", "max_PM", "afstand", "windrichting"]
PRED_LABELS = ["G_max", "M_max", "PM_max", "Afstand", "Windrichting\n(Gunstig/Ongunstig)"]

# Lettertype
FONT_PAD   = r"C:\Users\Gebruiker\AppData\Local\Microsoft\Windows\Fonts\UGentPannoText-Medium.ttf"

# ── 1. Data laden & voorbereiden ─────────────────────────────────────────────
df = pd.read_csv(INPUT_PATH, sep=";", decimal=",")

G_cols  = [c for c in df.columns if c.startswith("G_")]
M_cols  = [c for c in df.columns if c.startswith("M_")]
PM_cols = [c for c in df.columns if c.startswith("PM_")]
df["max_G"]  = df[G_cols].max(axis=1)
df["max_M"]  = df[M_cols].max(axis=1)
df["max_PM"] = df[PM_cols].max(axis=1)

gunstig = {'West', 'Zuidwest', 'Zuid'}
df['windrichting_bin'] = df['richting'].apply(lambda x: 1 if x in gunstig else 0)
df["stance_bin"] = (df["stance_numeriek"] != -1).astype(int)

data  = df[["max_G", "max_M", "max_PM", "afstand", "windrichting_bin", "stance_bin"]].dropna()
X_raw = data[["max_G", "max_M", "max_PM", "afstand", "windrichting_bin"]].values
y     = data["stance_bin"].values

print(f"N={len(y)}, Negatief={(y==0).sum()}, Niet-negatief={(y==1).sum()}")

def laad_font(font_pad):
    """Registreer UGent Panno Text als matplotlib-lettertype."""
    try:
        fm.fontManager.addfont(font_pad)
        prop = fm.FontProperties(fname=font_pad)
        familienaam = prop.get_name()
        plt.rcParams["font.family"] = familienaam
        print(f"Lettertype geladen: {familienaam}")
    except Exception as e:
        print(f"Lettertype niet gevonden ({e}). Teruggevallen op DejaVu Sans.")
        plt.rcParams["font.family"] = "DejaVu Sans"

# ── 2. Standaardiseren & PCA ─────────────────────────────────────────────────
scaler   = StandardScaler()
X_std    = scaler.fit_transform(X_raw)

pca_full = PCA()
pca_full.fit(X_std)
expl_var = pca_full.explained_variance_ratio_
cum_var  = np.cumsum(expl_var)
loadings = pca_full.components_   # shape: (5, 5)

# 2 componenten behouden (samen 72% verklaarde variantie)
N_COMP = 2
pca    = PCA(n_components=N_COMP)
X_pca  = pca.fit_transform(X_std)   # shape: (68, 2)

print(f"\nVerklaarde variantie per component: {np.round(pca.explained_variance_ratio_*100, 1)}%")
print(f"Totaal verklaard door {N_COMP} componenten: {pca.explained_variance_ratio_.sum()*100:.1f}%")

# ── 3. Logistische regressie op PCA-scores ───────────────────────────────────
X_sm     = sm.add_constant(X_pca)
model_sm = sm.Logit(y, X_sm).fit(disp=False)
print("\n", model_sm.summary())

params   = model_sm.params
pvals    = model_sm.pvalues
conf_np  = model_sm.conf_int()
coefs    = params[1:]
OR       = np.exp(coefs)
OR_low   = np.exp(conf_np[1:, 0])
OR_high  = np.exp(conf_np[1:, 1])
mcfadden = model_sm.prsquared
llr_pval = model_sm.llr_pvalue

clf      = LogisticRegression(random_state=42, max_iter=1000)
clf.fit(X_pca, y)
y_pred   = clf.predict(X_pca)
y_prob   = clf.predict_proba(X_pca)[:, 1]
fpr, tpr, _ = roc_curve(y, y_prob)
roc_auc  = auc(fpr, tpr)
cv_scores = cross_val_score(clf, X_pca, y, cv=5, scoring="accuracy")
accuracy  = (y_pred == y).mean()
null_acc  = max(y.mean(), 1 - y.mean())

# ── Plotinstellingen ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family"      : "Arial",
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "axes.grid"        : True,
    "grid.color"       : "#DDDDDD",
    "grid.linewidth"   : 0.6,
})

# ── Figuur 1: Scree plot + cumulatieve variantie ─────────────────────────────
laad_font(FONT_PAD)
fig, ax1 = plt.subplots(figsize=(7, 4.5))
x_pos = np.arange(1, 6)

bars = ax1.bar(x_pos, expl_var * 100, color=CLR_ACCENT, alpha=0.75,
               edgecolor="white", width=0.5, label="Verklaarde variantie per component")
ax1.set_xlabel("Principal Component", fontsize=10)
ax1.set_ylabel("Verklaarde variantie (%)", fontsize=10, color=CLR_ACCENT)
ax1.set_xticks(x_pos)
ax1.set_xticklabels([f"PC{i}" for i in x_pos])
ax1.set_ylim(0, 60)

ax2 = ax1.twinx()
ax2.plot(x_pos, cum_var * 100, color=CLR_NEG, marker="o",
         linewidth=2, markersize=7, label="Cumulatief")
ax2.axhline(72, color=CLR_NEG, linestyle="--", linewidth=1, alpha=0.5)
ax2.axvline(2.5, color="gray", linestyle="--", linewidth=1, alpha=0.4)
ax2.set_ylabel("Cumulatieve verklaarde variantie (%)", fontsize=10, color=CLR_NEG)
ax2.set_ylim(0, 110)

for i, v in enumerate(expl_var * 100):
    ax1.text(i + 1, v + 0.8, f"{v:.1f}%", ha="center", fontsize=8.5, color=CLR_ACCENT)

# Annotatievak voor geselecteerde componenten
ax1.add_patch(plt.Rectangle((0.55, 0), 2, 55, alpha=0.06,
              color=CLR_PC1, transform=ax1.transData, zorder=0))
ax1.text(1.5, 51, "Geselecteerd\n(PC1 + PC2)", ha="center", fontsize=8,
         color=CLR_PC1, fontstyle="italic")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="center right")
ax1.set_title("Scree plot — PCA op 5 predictoren", fontsize=11, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig1_scree_plot.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Figuur 1 opgeslagen.")

# ── Figuur 2: Loadingsplot (heatmap) ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
load_matrix = loadings[:N_COMP, :]   # (2, 5)

custom_cmap = LinearSegmentedColormap.from_list(
    'my_cmap',
    [   "#DD0D0DFC",
        '#ffffff',
        '#1e64c8',
    ]
)

ax.grid(False)
im = ax.imshow(load_matrix, cmap=custom_cmap, vmin=-1, vmax=1, aspect="auto")

cbar = plt.colorbar(im, ax=ax, label="Lading")
cbar.ax.tick_params(labelsize=12)
cbar.set_label("Lading", fontsize=13)

ax.set_xticks(range(5))
ax.set_xticklabels(PRED_LABELS, fontsize=11)
ax.set_yticks(range(N_COMP))
ax.set_yticklabels([f"PC{i+1}\n({expl_var[i]*100:.1f}%)" for i in range(N_COMP)], fontsize=11)

for i in range(N_COMP):
    for j in range(5):
        val = load_matrix[i, j]
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=15, color="white" if abs(val) > 0.5 else CLR_ACCENT,
                fontweight="bold" if abs(val) > 0.4 else "normal")

ax.set_title("PCA-ladingen — bijdrage van elke predictor per component",
             fontsize=16, fontweight="bold", pad=12)
fig.text(0.5, -0.02,
         "Blauwe waarden: positieve bijdrage  |  Rode waarden: negatieve bijdrage",
         ha="center", fontsize=10, color="gray")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig2_loadings_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Figuur 2 opgeslagen.")

# ── Figuur 3: Biplot (PC1 vs PC2, respondenten + loadingspijlen) ─────────────
fig, ax = plt.subplots(figsize=(7, 6))

colors_pts = [CLR_NEG if yi == 0 else CLR_POS for yi in y]
ax.scatter(X_pca[:, 0], X_pca[:, 1], c=colors_pts, alpha=0.6, s=45, zorder=2)

# Loadingspijlen
scale = 2.5
for j, (name, lbl) in enumerate(zip(PRED_NAMES, PRED_LABELS)):
    dx = loadings[0, j] * scale
    dy = loadings[1, j] * scale
    ax.annotate("", xy=(dx, dy), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=CLR_ACCENT, lw=1.8))
    offset_x = 0.08 if dx >= 0 else -0.08
    offset_y = 0.08 if dy >= 0 else -0.08
    ax.text(dx + offset_x, dy + offset_y,
            lbl.replace("\n", " "), fontsize=7.5, color=CLR_ACCENT,
            ha="center", va="center")

ax.axhline(0, color="gray", linewidth=0.8, alpha=0.5)
ax.axvline(0, color="gray", linewidth=0.8, alpha=0.5)
ax.set_xlabel(f"PC1 ({expl_var[0]*100:.1f}% verklaarde variantie)", fontsize=10)
ax.set_ylabel(f"PC2 ({expl_var[1]*100:.1f}% verklaarde variantie)", fontsize=10)
ax.set_title("Biplot — respondenten en predictoren in de PCA-ruimte",
             fontsize=11, fontweight="bold", pad=12)

patch0 = mpatches.Patch(color=CLR_NEG, label="Negatieve stance")
patch1 = mpatches.Patch(color=CLR_POS, label="Niet-negatieve stance")
ax.legend(handles=[patch0, patch1], fontsize=9, loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig3_biplot.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Figuur 3 opgeslagen.")

# ── Figuur 4: Odds Ratio's van de PCA-regressie ──────────────────────────────
fig, ax = plt.subplots(figsize=(7, 3.5))
y_pos   = np.arange(N_COMP)
pc_labels = [f"PC1\n({expl_var[0]*100:.1f}% var.)", f"PC2\n({expl_var[1]*100:.1f}% var.)"]
colors_or = [CLR_NEG if OR[i] < 1 else CLR_POS for i in range(N_COMP)]

for i in range(N_COMP):
    ax.plot([OR_low[i], OR_high[i]], [y_pos[i], y_pos[i]],
            color=colors_or[i], linewidth=2.5, zorder=2)
    ax.scatter(OR[i], y_pos[i], color=colors_or[i], s=90, zorder=3)
    sig = "**" if pvals[i+1] < 0.01 else ("*" if pvals[i+1] < 0.05 else "")
    ax.text(OR_high[i] + 0.04, y_pos[i],
            f"OR={OR[i]:.2f}{sig}  p={pvals[i+1]:.3f}",
            va="center", fontsize=9, color=CLR_ACCENT)

ax.axvline(1, color="gray", linestyle="--", linewidth=1.2)
ax.set_yticks(y_pos)
ax.set_yticklabels(pc_labels, fontsize=10)
ax.set_xlabel("Odds Ratio (met 95%-BI)", fontsize=10)
ax.set_title("Odds Ratio's — PCA-regressiemodel\n(ongecorreleerde componenten als predictoren)",
             fontsize=11, fontweight="bold", pad=12)
ax.set_xlim(left=max(0, OR_low.min() - 0.3))
fig.text(0.5, -0.02, "* p < 0,05   ** p < 0,01   Referentie: OR = 1 (geen effect)",
         ha="center", fontsize=8, color="gray")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig4_odds_ratios.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Figuur 4 opgeslagen.")

# ── Figuur 5: ROC-curve ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.5, 5.5))
ax.plot(fpr, tpr, color=CLR_ACCENT, lw=2,
        label=f"ROC-curve (AUC = {roc_auc:.2f})")
ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1.2, label="Kanslijn")
ax.set_xlabel("Vals-positief rate (1 – specificiteit)", fontsize=10)
ax.set_ylabel("Echt-positief rate (sensitiviteit)", fontsize=10)
ax.set_title("ROC-curve — PCA-regressiemodel", fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig5_roc_curve.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Figuur 5 opgeslagen.")

# ── Figuur 6: Verwarringsmatrix ──────────────────────────────────────────────
cm  = confusion_matrix(y, y_pred)
fig, ax = plt.subplots(figsize=(5, 4))
disp = ConfusionMatrixDisplay(cm, display_labels=["Negatief", "Niet-negatief"])
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("Verwarringsmatrix — PCA-regressiemodel", fontsize=11, fontweight="bold", pad=10)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig6_verwarringsmatrix.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Figuur 6 opgeslagen.")

# ── Figuur 7: Scatter PC1 vs PC2 met beslissingsgrens ────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))

# Beslissingsgrens
x_min, x_max = X_pca[:, 0].min() - 0.5, X_pca[:, 0].max() + 0.5
y_min, y_max = X_pca[:, 1].min() - 0.5, X_pca[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))
Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)
ax.contourf(xx, yy, Z, levels=50, cmap="RdBu_r", alpha=0.25)
ax.contour(xx, yy, Z, levels=[0.5], colors=CLR_ACCENT, linewidths=1.8,
           linestyles="--")

# Respondenten
ax.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1],
           color=CLR_NEG, edgecolors="white", s=55, alpha=0.85,
           label="Negatieve stance", zorder=3)
ax.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1],
           color=CLR_POS, edgecolors="white", s=55, alpha=0.85,
           label="Niet-negatieve stance", zorder=3)

ax.set_xlabel(f"PC1 ({expl_var[0]*100:.1f}% verklaarde variantie)", fontsize=10)
ax.set_ylabel(f"PC2 ({expl_var[1]*100:.1f}% verklaarde variantie)", fontsize=10)
ax.set_title("Beslissingsgrens van het PCA-regressiemodel\nin de PCA-ruimte",
             fontsize=11, fontweight="bold", pad=12)
ax.legend(fontsize=9)
fig.text(0.5, -0.02,
         "Stippellijn = beslissingsgrens (P = 0,5)  |  Kleurgradiënt = voorspelde kans",
         ha="center", fontsize=8, color="gray")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig7_beslissingsgrens.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Figuur 7 opgeslagen.")

# ── Figuur 8: Modelfit tabel ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4.2))
ax.axis("off")
metrics = [
    ("N respondenten",            f"{len(y)}"),
    ("Events (negatief)",          f"{(y==0).sum()}"),
    ("Aantal componenten (PCA)",   f"{N_COMP}"),
    ("Totaal verklaarde variantie",f"{pca.explained_variance_ratio_.sum()*100:.1f}%"),
    ("EPV",                        f"{(y==0).sum()/N_COMP:.1f}"),
    ("McFadden pseudo-R²",         f"{mcfadden:.3f}"),
    ("LLR p-waarde",               f"{llr_pval:.4f}"),
    ("AUC",                        f"{roc_auc:.3f}"),
    ("Accuraatheid (in-sample)",   f"{accuracy:.3f}"),
    ("Null-accuraatheid",          f"{null_acc:.3f}"),
    ("CV accuraatheid (5-fold)",   f"{cv_scores.mean():.3f} ± {cv_scores.std():.3f}"),
]
for row_i, (label, val) in enumerate(metrics):
    bg = CLR_LIGHT if row_i % 2 == 0 else "white"
    ax.add_patch(plt.Rectangle((0, row_i/len(metrics)), 1, 1/len(metrics),
                                transform=ax.transAxes, color=bg, zorder=0))
    ax.text(0.05, (row_i + 0.5)/len(metrics), label,
            transform=ax.transAxes, fontsize=10, va="center")
    ax.text(0.72, (row_i + 0.5)/len(metrics), val,
            transform=ax.transAxes, fontsize=10, va="center", fontweight="bold")
ax.set_title("Modelfitstatistieken — PCA-regressiemodel",
             fontsize=11, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig8_modelfit.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Figuur 8 opgeslagen.")

print("\n✓ Alle figuren PCA-regressie opgeslagen in:", OUTPUT_DIR)