import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from scipy import stats
import warnings, os

warnings.filterwarnings('ignore')

# ── Paden ────────────────────────────────────────────────────────────────────
INPUT_PATH = r"C:\Users\Gebruiker\Documents\DATA\Resultaten\DATASET_zonderGSNB.csv"
OUTPUT_DIR = r"C:\Users\Gebruiker\Documents\DATA\Grafieken\Regressie_model3"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Lettertype
FONT_PAD   = r"C:\Users\Gebruiker\AppData\Local\Microsoft\Windows\Fonts\UGentPannoText-Medium.ttf"

CLR_NEG    = "#C0392B"
CLR_POS    = "#2980B9"
CLR_ACCENT = "#2C3E50"
CLR_LIGHT  = "#ECF0F1"

# ── 1. Data laden & voorbereiden ─────────────────────────────────────────────
df = pd.read_csv(INPUT_PATH, sep=";", decimal=",")

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

G_cols  = [c for c in df.columns if c.startswith("G_")]
M_cols  = [c for c in df.columns if c.startswith("M_")]
PM_cols = [c for c in df.columns if c.startswith("PM_")]
df["max_G"]  = df[G_cols].max(axis=1)
df["max_M"]  = df[M_cols].max(axis=1)
df["max_PM"] = df[PM_cols].max(axis=1)

gunstig = {'West', 'Zuidwest', 'Zuid'}
df['windrichting_bin'] = df['richting'].apply(lambda x: 1 if x in gunstig else 0)
df["stance_bin"] = (df["stance_numeriek"] != -1).astype(int)

data = df[["max_G", "max_M", "max_PM", "afstand", "windrichting_bin", "stance_bin"]].dropna()
y    = data["stance_bin"].values

# Standaardiseer continue variabelen; windrichting (binair) niet
scaler  = StandardScaler()
cont_cols = ["max_G", "max_M", "max_PM", "afstand"]
X_cont_std = scaler.fit_transform(data[cont_cols].values)
X_raw  = np.column_stack([X_cont_std, data["windrichting_bin"].values])

predictor_labels = [
    "Max. gem.\ngeluidsniveau (LAeq)",
    "Max. max.\ngeluidsniveau (LAmax)",
    "Max. aantal\ngeluidspieken",
    "Afstand\n(gestandaardiseerd)",
    "Windrichting\n(Gunstig vs. Ongunstig)"
]

print(f"N respondenten    : {len(y)}")
print(f"Negatief (0)      : {(y==0).sum()}")
print(f"Niet-negatief (1) : {(y==1).sum()}")
print(f"EPV               : {(y==0).sum() / X_raw.shape[1]:.1f}  ← onder aanbevolen grens van 10")

# ── 2. Statsmodels ───────────────────────────────────────────────────────────
X_sm     = sm.add_constant(X_raw)
model_sm = sm.Logit(y, X_sm).fit(disp=False)
print("\n", model_sm.summary())

params   = model_sm.params
pvals_np = model_sm.pvalues
conf_np  = model_sm.conf_int()

coefs   = params[1:]
pvals   = pvals_np[1:]
OR      = np.exp(coefs)
OR_low  = np.exp(conf_np[1:, 0])
OR_high = np.exp(conf_np[1:, 1])
n_pred  = len(coefs)
mcfadden = model_sm.prsquared
llr_pval = model_sm.llr_pvalue
vif_data = pd.DataFrame()
vif_data["Predictor"] = ["max_G", "max_M", "max_PM", "afstand", "windrichting"]
vif_data["VIF"] = [variance_inflation_factor(X_raw, i) for i in range(X_raw.shape[1])]
print(vif_data)

# ── 3. Sklearn ───────────────────────────────────────────────────────────────
clf      = LogisticRegression(random_state=42, max_iter=1000)
clf.fit(X_raw, y)
y_pred   = clf.predict(X_raw)
y_prob   = clf.predict_proba(X_raw)[:, 1]
fpr, tpr, _ = roc_curve(y, y_prob)
roc_auc  = auc(fpr, tpr)
cv_scores = cross_val_score(clf, X_raw, y, cv=5, scoring="accuracy")

# ── Plotinstellingen ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family"      : "Arial",
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "axes.grid"        : True,
    "grid.color"       : "#DDDDDD",
    "grid.linewidth"   : 0.6,
})

# ── Figuur 1: Odds Ratio plot ────────────────────────────────────────────────
laad_font(FONT_PAD)
fig, ax = plt.subplots(figsize=(8, 4.5))
colors = [CLR_NEG if OR[i] < 1 else CLR_POS for i in range(n_pred)]
y_pos  = np.arange(n_pred)

for i in range(n_pred):
    ax.plot([OR_low[i], OR_high[i]], [y_pos[i], y_pos[i]],
            color=colors[i], linewidth=2.5, zorder=2)
    ax.scatter(OR[i], y_pos[i], color=colors[i], s=90, zorder=3)
    sig = "**" if pvals[i] < 0.01 else ("*" if pvals[i] < 0.05 else "")
    ax.text(OR_high[i] + 0.05, y_pos[i],
            f"OR={OR[i]:.2f}{sig}  p={pvals[i]:.3f}",
            va="center", fontsize=12, color=CLR_ACCENT)

ax.axvline(1, color="gray", linestyle="--", linewidth=1.2)
ax.set_yticks(y_pos)
ax.set_yticklabels(predictor_labels, fontsize=12)
ax.set_xlabel("Odds Ratio (met 95%-BI)", fontsize=15)
ax.tick_params(axis='x', labelsize=12)
ax.set_title("Odds Ratio's — Logistisch regressiemodel 3\n(volledig model)",
             fontsize=20, fontweight="bold", pad=12)
ax.set_xlim(left=max(0, OR_low.min() - 0.3))
fig.text(0.5, -0.02,
         "* p < 0,05   ** p < 0,01   Referentie windrichting: Ongunstig  |  EPV = 7,2 (verkennend)",
         ha="center", fontsize=12, color="gray")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig1_odds_ratios.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Figuur 1 opgeslagen.")

# ── Figuur 2: ROC-curve ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.5, 5.5))
ax.plot(fpr, tpr, color=CLR_ACCENT, lw=2,
        label=f"ROC-curve (AUC = {roc_auc:.2f})")
ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1.2, label="Kanslijn")
ax.set_xlabel("Vals-positief rate (1 – specificiteit)", fontsize=10)
ax.set_ylabel("Echt-positief rate (sensitiviteit)", fontsize=10)
ax.set_title("ROC-curve — Model 3", fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig2_roc_curve.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Figuur 2 opgeslagen.")

# ── Figuur 3: Verwarringsmatrix ──────────────────────────────────────────────
cm   = confusion_matrix(y, y_pred)
fig, ax = plt.subplots(figsize=(5, 4))
disp = ConfusionMatrixDisplay(cm, display_labels=["Negatief", "Niet-negatief"])
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("Verwarringsmatrix — Model 3", fontsize=11, fontweight="bold", pad=10)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig3_verwarringsmatrix.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Figuur 3 opgeslagen.")

# ── Figuur 4: Boxplots alle predictoren ─────────────────────────────────────
cont_labels = [
    "Max. gem. geluidsniveau\n(LAeq, dBA)",
    "Max. max. geluidsniveau\n(LAmax, dBA)",
    "Max. aantal\ngeluidspieken",
    "Afstand tot podium (m)"
]
cont_raw = ["max_G", "max_M", "max_PM", "afstand"]

fig, axes = plt.subplots(1, 5, figsize=(18, 5))

# Boxplots voor 4 continue predictoren
for i, (col, lbl) in enumerate(zip(cont_raw, cont_labels)):
    ax = axes[i]
    grp0 = data[data["stance_bin"] == 0][col]
    grp1 = data[data["stance_bin"] == 1][col]
    bp = ax.boxplot([grp0, grp1], patch_artist=True,
                    medianprops=dict(color="white", linewidth=2),
                    whiskerprops=dict(linewidth=1.2), capprops=dict(linewidth=1.2),
                    flierprops=dict(marker="o", markersize=4, alpha=0.5))
    bp["boxes"][0].set_facecolor(CLR_NEG)
    bp["boxes"][1].set_facecolor(CLR_POS)
    stat, p = stats.mannwhitneyu(grp0, grp1, alternative="two-sided")
    sig_str = f"p = {p:.3f}" + (" *" if p < 0.05 else "")
    ax.set_title(f"{lbl}\n({sig_str})", fontsize=8.5, pad=8)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Neg.", "Niet-neg."], fontsize=8)

# Staafdiagram windrichting
ax = axes[4]
ct = pd.crosstab(data["windrichting_bin"], data["stance_bin"])
ct.index = ["Ongunstig", "Gunstig"]
ct.columns = ["Negatief", "Niet-negatief"]
ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
ct_pct.plot(kind="bar", ax=ax, color=[CLR_NEG, CLR_POS],
            edgecolor="white", width=0.5)
chi2, p_chi, _, _ = stats.chi2_contingency(ct)
sig_str = f"p = {p_chi:.3f}" + (" *" if p_chi < 0.05 else "")
ax.set_title(f"Windrichting\n(χ² {sig_str})", fontsize=8.5, pad=8)
ax.set_xlabel("")
ax.set_xticklabels(["Ongunstig", "Gunstig"], rotation=0, fontsize=8)
ax.set_ylabel("%", fontsize=8)
ax.set_ylim(0, 100)
ax.legend().remove()

patch0 = mpatches.Patch(color=CLR_NEG, label="Negatieve stance")
patch1 = mpatches.Patch(color=CLR_POS, label="Niet-negatieve stance")
fig.legend(handles=[patch0, patch1], loc="lower center", ncol=2,
           fontsize=9, bbox_to_anchor=(0.5, -0.04))
fig.suptitle("Verdeling predictoren per stancegroep — Model 3",
             fontsize=12, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig4_boxplots.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Figuur 4 opgeslagen.")

# ── Figuur 5: Marginale kansen ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 5, figsize=(18, 5))
means_all = data[["max_G", "max_M", "max_PM", "afstand"]].mean().values
mean_wind = data["windrichting_bin"].mean()

for i, (col, lbl) in enumerate(zip(cont_raw, cont_labels)):
    ax = axes[i]
    x_range = np.linspace(data[col].min(), data[col].max(), 200)
    X_sim_cont = np.tile(means_all, (200, 1))
    X_sim_cont[:, i] = x_range
    X_sim_cont_std = scaler.transform(X_sim_cont)
    X_sim = np.column_stack([X_sim_cont_std, np.full(200, mean_wind)])
    probs = clf.predict_proba(X_sim)[:, 1]
    jitter = np.random.default_rng(42).uniform(-0.01, 0.01, len(data))
    ax.scatter(data[col], y + jitter,
               c=[CLR_NEG if yi == 0 else CLR_POS for yi in y],
               alpha=0.45, s=20, zorder=2)
    ax.plot(x_range, probs, color=CLR_ACCENT, lw=2.2, zorder=3)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel(lbl, fontsize=8)
    ax.set_ylabel("P(niet-negatief)", fontsize=8)
    ax.set_ylim(-0.08, 1.08)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])

# Windrichting: puntplot
ax = axes[4]
means_cont_std = scaler.transform([means_all])[0]
for val, label, clr, xpos in [(0, "Ongunstig", CLR_NEG, 0), (1, "Gunstig", CLR_POS, 1)]:
    X_pt  = np.array([np.append(means_cont_std, val)])
    prob  = clf.predict_proba(X_pt)[0, 1]
    ax.scatter(xpos, prob, color=clr, s=120, zorder=3)
ax.set_xticks([0, 1])
ax.set_xticklabels(["Ongunstig", "Gunstig"], fontsize=8)
ax.set_ylabel("P(niet-negatief)", fontsize=8)
ax.set_title("Windrichting", fontsize=8.5)
ax.set_ylim(0, 1)
ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7)

patch0 = mpatches.Patch(color=CLR_NEG, label="Negatieve stance")
patch1 = mpatches.Patch(color=CLR_POS, label="Niet-negatieve stance")
fig.legend(handles=[patch0, patch1], loc="lower center", ncol=2,
           fontsize=9, bbox_to_anchor=(0.5, -0.04))
fig.suptitle("Marginale voorspelde kansen per predictor — Model 3\n(overige predictoren op gemiddelde)",
             fontsize=11, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig5_marginale_kansen.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Figuur 5 opgeslagen.")

# ── Figuur 6: Modelfit tabel ─────────────────────────────────────────────────
accuracy = (y_pred == y).mean()
null_acc = max(y.mean(), 1 - y.mean())
fig, ax  = plt.subplots(figsize=(6, 4.2))
ax.axis("off")
metrics = [
    ("N respondenten",            f"{len(y)}"),
    ("Events (negatief)",          f"{(y==0).sum()}"),
    ("EPV",                        f"{(y==0).sum()/n_pred:.1f}  ⚠ verkennend"),
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
    ax.text(0.70, (row_i + 0.5)/len(metrics), val,
            transform=ax.transAxes, fontsize=10, va="center", fontweight="bold")
ax.set_title("Modelfitstatistieken — Model 3  (EPV < 10: verkennend)",
             fontsize=11, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig6_modelfit.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Figuur 6 opgeslagen.")

print("\n✓ Alle figuren Model 3 opgeslagen in:", OUTPUT_DIR)