import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from scipy.stats import false_discovery_control
import warnings
warnings.filterwarnings('ignore')

# ── Stijlinstellingen ────────────────────────────────────────────────────────
KLEUR_NEG   = "#C0392B"
KLEUR_POS   = "#2980B9"
KLEUR_GEM   = "#79d1ff"
KLEUR_MAX   = "#1e64c8"
ACHTERGROND = "#FFFFFF"
RASTER      = "#DDE3EA"

# Lettertype
FONT_PAD   = r"C:\Users\Gebruiker\AppData\Local\Microsoft\Windows\Fonts\UGentPannoText-Medium.ttf"

plt.rcParams.update({
    "figure.facecolor":  ACHTERGROND,
    "axes.facecolor":    ACHTERGROND,
    "axes.edgecolor":    "#BDC3C7",
    "axes.grid":         True,
    "grid.color":        RASTER,
    "grid.linewidth":    0.8,
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.framealpha": 0.9,
    "legend.edgecolor":  "#BDC3C7",
})

# ── 1. Data inladen & opschonen ──────────────────────────────────────────────
df = pd.read_csv('C:\\Users\\Gebruiker\\Documents\\DATA\\Resultaten\\DATASET_zonderGSNB.csv', sep=';')

df = df.loc[:, ~df.columns.str.startswith('Unnamed')]

for col in df.columns:
    if col.startswith('G_') or col.startswith('M_'):
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(',', '.', regex=False), errors='coerce')
    elif col.startswith('PI_') or col.startswith('PM_'):
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(',', '.', regex=False), errors='coerce')

# ── 2. Binaire stance ────────────────────────────────────────────────────────
df['stance_binair'] = (df['stance_numeriek'] != -1).astype(int)

# ── 3. Aggregatie ────────────────────────────────────────────────────────────
metrieken = {
    'G':  [c for c in df.columns if c.startswith('G_')  and '29aug' not in c],
    'M':  [c for c in df.columns if c.startswith('M_')  and '29aug' not in c],
    'PI': [c for c in df.columns if c.startswith('PI_') and '29aug' not in c],
    'PM': [c for c in df.columns if c.startswith('PM_') and '29aug' not in c],
}

labels_nl = {
    'G':  'Gemiddeld geluidsniveau (G)',
    'M':  'Maximaal geluidsniveau (M)',
    'PI': 'Percentage boven drempel (PI)',
    'PM': 'Piekaantal overschrijdingen (PM)',
}

titels_officieel = {
    'G':  'Gemiddelde',
    'M':  'Maximaal',
    'PI': 'Piekintensiteit',
    'PM': 'Aantal pieken',
}

for prefix, cols in metrieken.items():
    df[f'{prefix}_gem'] = df[cols].mean(axis=1, skipna=True)
    df[f'{prefix}_max'] = df[cols].max(axis=1, skipna=True)

# ── 4. Correlatieberekeningen ────────────────────────────────────────────────
resultaten = []

for prefix in metrieken:
    for aggr, aggr_label in [('gem', 'Gemiddelde'), ('max', 'Maximum')]:
        kolom = f'{prefix}_{aggr}'
        x = df[kolom]
        y = df['stance_binair']
        mask = x.notna() & y.notna()
        x_clean = x[mask]
        y_clean = y[mask]

        pb_r, pb_p = stats.pointbiserialr(y_clean, x_clean)
        tau, tau_p = stats.kendalltau(x_clean, y_clean)

        resultaten.append({
            'Metriek':       prefix,
            'Metriek_label': labels_nl[prefix],
            'Aggregatie':    aggr_label,
            'Kolom':         kolom,
            'PB_r':          pb_r,
            'PB_p':          pb_p,
            'Kendall_tau':   tau,
            'Kendall_p':     tau_p,
            'n':             mask.sum(),
        })

res = pd.DataFrame(resultaten)

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

# ── 5. Correcties & effectgroottes ───────────────────────────────────────────
alpha = 0.05
bonferroni_drempel = alpha / len(res)

res['PB_sig']             = res['PB_p']     < bonferroni_drempel
res['Kendall_sig']        = res['Kendall_p'] < bonferroni_drempel
res['PB_sig_ongecorr']    = res['PB_p']     < alpha
res['Kendall_sig_ongecorr'] = res['Kendall_p'] < alpha

res['PB_sig_BH']      = false_discovery_control(res['PB_p'].values)  < alpha
res['Kendall_sig_BH'] = false_discovery_control(res['Kendall_p'].values) < alpha

# Effectgroottes: r² en Cohen's d
cohens_d_lijst = []
for _, row in res.iterrows():
    kolom = row['Kolom']
    grp0 = df[df['stance_binair'] == 0][kolom].dropna()
    grp1 = df[df['stance_binair'] == 1][kolom].dropna()
    pooled_sd = np.sqrt(((len(grp0)-1)*grp0.std()**2 + (len(grp1)-1)*grp1.std()**2)
                        / (len(grp0) + len(grp1) - 2))
    d = (grp1.mean() - grp0.mean()) / pooled_sd if pooled_sd > 0 else 0
    cohens_d_lijst.append(d)

res['r_kwadraat'] = res['PB_r'] ** 2
res['cohens_d']   = cohens_d_lijst

# ── 6. Resultaten printen ────────────────────────────────────────────────────
print("=" * 72)
print("CORRELATIEANALYSE — Geluidsmetrieken vs. Stance (negatief / niet-negatief)")
print("=" * 72)
print(f"Bonferroni-drempel: α = {bonferroni_drempel:.4f} (0.05 / {len(res)} toetsen)")
print(f"BH (FDR)-drempel:   adaptief per rang (α = {alpha})")
print()

for _, row in res.iterrows():
    sig_pb  = "✓ Bonf." if row['PB_sig'] else ("~ BH" if row['PB_sig_BH'] else ("~ ongecorr." if row['PB_sig_ongecorr'] else "✗ n.s."))
    sig_tau = "✓ Bonf." if row['Kendall_sig'] else ("~ BH" if row['Kendall_sig_BH'] else ("~ ongecorr." if row['Kendall_sig_ongecorr'] else "✗ n.s."))
    print(f"  {row['Metriek_label']} [{row['Aggregatie']}]")
    print(f"    Punt-biserieel r = {row['PB_r']:+.3f}  r² = {row['r_kwadraat']:.3f}  Cohen's d = {row['cohens_d']:+.3f}  (p = {row['PB_p']:.4f})  {sig_pb}")
    print(f"    Kendall's τ      = {row['Kendall_tau']:+.3f}  (p = {row['Kendall_p']:.4f})  {sig_tau}")
    print()

# ── 7. FIGUUR 1 — Correlatiesterkte overzicht ────────────────────────────────
laad_font(FONT_PAD)
labels_as = {
    'G':  'Gemiddelde',
    'M':  'Maximum',
    'PI': 'Piekintensiteit',
    'PM': 'Aantal pieken',
}

fig1, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
fig1.suptitle("Correlatiecoëfficiënten: Geluidsmetrieken vs. Stance\n(negatief vs. niet-negatief)",
              fontsize=22, fontweight='bold', y=1.01)

x_pos   = np.arange(len(metrieken))
breedte = 0.35
metriek_keys = list(metrieken.keys())

for ax_idx, (toets_label, r_col, p_col, sig_col) in enumerate([
    ("Punt-biseriële correlatie (r)", "PB_r",       "PB_p",       "PB_sig_ongecorr"),
    ("Kendall's τ",                   "Kendall_tau", "Kendall_p",  "Kendall_sig_ongecorr"),
]):
    ax = axes[ax_idx]
    ax.set_axisbelow(True)

    for i, key in enumerate(metriek_keys):
        rij_gem = res[(res['Metriek'] == key) & (res['Aggregatie'] == 'Gemiddelde')].iloc[0]
        rij_max = res[(res['Metriek'] == key) & (res['Aggregatie'] == 'Maximum')].iloc[0]

        val_gem = rij_gem[r_col]
        val_max = rij_max[r_col]
        sig_gem = rij_gem[sig_col]
        sig_max = rij_max[sig_col]

        bar1 = ax.bar(i - breedte/2, val_gem, breedte,
                      color=KLEUR_GEM, alpha=0.85, label='Gemiddelde' if i == 0 else '', zorder=3)
        bar2 = ax.bar(i + breedte/2, val_max, breedte,
                      color=KLEUR_MAX, alpha=0.85, label='Maximum' if i == 0 else '', zorder=3)

        for bar, val, sig in [(bar1, val_gem, sig_gem), (bar2, val_max, sig_max)]:
            if sig:
                offset = 0.01 if val >= 0 else -0.03
                ax.text(bar[0].get_x() + bar[0].get_width()/2,
                        val + offset, '*', ha='center', va='bottom',
                        fontsize=18, color='#E74C3C', fontweight='bold', zorder=4)

        coef_tekst = f"r={val_gem:+.3f}\nr={val_max:+.3f}" if r_col == 'PB_r' else f"τ={val_gem:+.3f}\nτ={val_max:+.3f}"
        ax.text(i, -0.40, coef_tekst, ha='center', va='top',
                fontsize=18, color='#2C3E50', linespacing=1.6)

    ax.axhline(0, color='#7F8C8D', linewidth=1.0, linestyle='--', zorder=2)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([labels_as[k] for k in metriek_keys], fontsize=17)
    ax.set_title(toets_label, fontsize=18, fontweight='bold')
    ax.set_ylabel("Correlatiecoëfficiënt", fontsize=18)
    ax.tick_params(axis='y', labelsize=17)
    ax.set_ylim(-0.55, 0.5)
    ax.legend(loc='upper right', fontsize=17)
    ax.text(0.02, 0.97, "* = p < 0.05 (ongecorrigeerd)",
            transform=ax.transAxes, fontsize=17, va='top', color='#E74C3C')

plt.tight_layout()
plt.savefig('C:\\Users\\Gebruiker\\Documents\\DATA\\Grafieken\\Correlatie\\1_correlatie_overzicht.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figuur 1 opgeslagen.")

# ── 8. FIGUUR 2 — P-waarden overzicht ───────────────────────────────────────
fig5, ax5 = plt.subplots(figsize=(12, 6))
fig5.suptitle("P-waarden per metriek en aggregatie", fontsize=23, fontweight='bold')
ax5.set_axisbelow(True)

x_labels, pb_pvals, tau_pvals = [], [], []
for _, rij in res.iterrows():
    x_labels.append(f"{rij['Metriek']}\n[{rij['Aggregatie'][:3]}]")
    pb_pvals.append(rij['PB_p'])
    tau_pvals.append(rij['Kendall_p'])

x = np.arange(len(x_labels))
breedte = 0.35

bars_pb  = ax5.bar(x - breedte/2, pb_pvals,  breedte, label="Punt-biserieel", color=KLEUR_GEM, alpha=0.8, zorder=3)
bars_tau = ax5.bar(x + breedte/2, tau_pvals, breedte, label="Kendall's τ",    color=KLEUR_MAX, alpha=0.8, zorder=3)
ax5.axhline(alpha,              color='#E74C3C', linewidth=1.8, linestyle='--', zorder=2,
            label=f"α = {alpha} (ongecorrigeerd)")
ax5.axhline(bonferroni_drempel, color='#8E44AD', linewidth=1.8, linestyle=':', zorder=2,
            label=f"Bonferroni = {bonferroni_drempel:.4f}")

# P-waarden boven de staven
for bar, val in zip(bars_pb, pb_pvals):
    ax5.text(bar.get_x() + bar.get_width() / 2,
             val + 0.008,
             f"{val:.3f}",
             ha='center', va='bottom', fontsize=15,
             color='#2C3E50', fontweight='bold' if val < alpha else 'normal')

for bar, val in zip(bars_tau, tau_pvals):
    ax5.text(bar.get_x() + bar.get_width() / 2,
             val + 0.008,
             f"{val:.3f}",
             ha='center', va='bottom', fontsize=15,
             color='#2C3E50', fontweight='bold' if val < alpha else 'normal')

ax5.set_xticks(x)
ax5.set_xticklabels(x_labels, fontsize=15)
ax5.set_ylabel("p-waarde", fontsize=18)
ax5.tick_params(axis='y', labelsize=15)
ax5.set_ylim(0, 0.68)
ax5.legend(fontsize=15)

plt.tight_layout()
plt.savefig('C:\\Users\\Gebruiker\\Documents\\DATA\\Grafieken\\Correlatie\\2_pwaarden.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figuur 2 opgeslagen.")

# ── 9. FIGUUR 3 — Multicollineariteit ───────────────────────────────────────
agg_kolommen  = [f'{p}_{a}' for p in metriek_keys for a in ['gem', 'max']]
labels_matrix = [f'{p}\n[{"Gem." if a == "gem" else "Max."}]'
                 for p in metriek_keys for a in ['gem', 'max']]

corr_matrix = df[agg_kolommen].corr(method='spearman')

fig7, ax7 = plt.subplots(figsize=(11, 9))
ax7.set_axisbelow(True)

custom_cmap = LinearSegmentedColormap.from_list(
    'my_cmap',
    [   '#e36c0a',
        '#fbd4b4',
        '#ffffff',
        '#d0e0f8',
        '#79d1ff',
    ]
)

im7 = ax7.imshow(corr_matrix.values, cmap=custom_cmap, vmin=-1, vmax=1, aspect='auto')

ax7.set_xticks(np.arange(-0.5, len(agg_kolommen), 1), minor=True)
ax7.set_yticks(np.arange(-0.5, len(agg_kolommen), 1), minor=True)
ax7.grid(which='minor', color='#7F8C8D', linewidth=1.2)
ax7.tick_params(which='minor', bottom=False, left=False)
ax7.grid(which='major', visible=False)
ax7.spines[:].set_visible(False)

ax7.set_xticks(range(len(agg_kolommen)))
ax7.set_yticks(range(len(agg_kolommen)))
ax7.set_xticklabels(labels_matrix, fontsize=15, rotation=30, ha='right')
ax7.set_yticklabels(labels_matrix, fontsize=15)
ax7.set_title("Multicollineariteit: Spearman-correlaties tussen geluidsmetrieken\n",
              fontsize=22, fontweight='bold')

for i in range(len(agg_kolommen)):
    for j in range(len(agg_kolommen)):
        val = corr_matrix.values[i, j]
        ax7.text(j, i, f"{val:.2f}", ha='center', va='center',
                 fontsize=15, fontweight='bold' if abs(val) > 0.7 else 'normal',
                 color='#2C3E50')

cbar = plt.colorbar(im7, ax=ax7, shrink=1, label="Spearman ρ")
cbar.ax.tick_params(labelsize=15)
cbar.set_label("Spearman ρ", fontsize=18)
plt.tight_layout()
plt.savefig('C:\\Users\\Gebruiker\\Documents\\DATA\\Grafieken\\Correlatie\\3_multicollineariteit.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figuur 3 opgeslagen.")

print()
print("=" * 72)
print("Alle figuren en analyse voltooid.")
print("=" * 72)