import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import wilcoxon
from itertools import combinations
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# KLEURINSTELLINGEN
# =============================================================================

KLEUREN = {
    "29aug_blok1": "#fbd4b4",   
    "29aug_blok2": "#fbd4b4",   
    "30aug_blok1": "#79d1ff",  
    "30aug_blok2": "#79d1ff",   
    "30aug_blok3": "#79d1ff",   
    "30aug_blok4": "#79d1ff",  
    "31aug_blok1": "#92d050",   
    "31aug_blok2": "#92d050",   
    "31aug_blok3": "#92d050",   
    "31aug_blok4": "#92d050",   
}

ACHTERGROND_KLEUR = "#FFFFFF"   
SIGNIFICANTIE_LIJN = "#CE0909"  
DREMPEL_ALPHA = 0.05  

# Lettertype
FONT_PAD   = r"C:\Users\Gebruiker\AppData\Local\Microsoft\Windows\Fonts\UGentPannoText-Medium.ttf"
# =============================================================================
# STAP 1 — Data inladen en voorbereiden
# =============================================================================


df = pd.read_csv(
    'C:/Users/Gebruiker/Documents/DATA/Resultaten/DATASET.csv',
    sep=';',
    encoding='utf-8-sig',
    decimal=','
)

# De 10 blokkolommen in chronologische volgorde
blok_kolommen = [
    'G_29aug_blok1', 'G_29aug_blok2',
    'G_30aug_blok1', 'G_30aug_blok2', 'G_30aug_blok3', 'G_30aug_blok4',
    'G_31aug_blok1', 'G_31aug_blok2', 'G_31aug_blok3', 'G_31aug_blok4',
]

# Leesbare labels voor de grafieken
blok_labels = [
    "29/08 B1", "29/08 B2",
    "30/08 B1", "30/08 B2", "30/08 B3", "30/08 B4",
    "31/08 B1", "31/08 B2", "31/08 B3", "31/08 B4",
]

# Selecteer enkel de rijen waar alle 10 blokken een geldige waarde hebben
# (geen NaN), zodat de Friedman-test op een complete matrix werkt.
data_blokken = df[blok_kolommen].apply(pd.to_numeric, errors='coerce')
data_compleet = data_blokken.dropna()

n_locaties = len(data_compleet)
print(f"Aantal locaties met volledige data: {n_locaties} / {len(df)}")

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

# =============================================================================
# STAP 2 — Friedman-test
# =============================================================================
# De Friedman-test vergelijkt k gerelateerde groepen (hier: 10 blokken) zonder
# normaliteitsassumptie. Per rij (= per locatie) worden de 10 waarden gerangschikt.
# De teststatistiek meet of de rangschikkingen systematisch van elkaar afwijken.
#
# H0: de medianen van alle 10 blokken zijn gelijk
# H1: minstens één blok wijkt significant af

friedman_stat, friedman_p = stats.friedmanchisquare(
    *[data_compleet[k].values for k in blok_kolommen]
)

n_blokken = len(blok_kolommen)
W = friedman_stat / (n_locaties * (n_blokken - 1))
print(f"Kendall's W = {W:.3f}")
print("\n--- FRIEDMAN-TEST (alle 10 blokken) ---")
print(f"Teststatistiek (χ²): {friedman_stat:.4f}")
print(f"p-waarde:            {friedman_p:.6f}")
if friedman_p < DREMPEL_ALPHA:
    print(f"→ SIGNIFICANT (p < {DREMPEL_ALPHA}): minstens één blok wijkt af")
else:
    print(f"→ Niet significant (p ≥ {DREMPEL_ALPHA})")

# =============================================================================
# STAP 3 — Post-hoc paarsgewijze Wilcoxon signed-rank tests
# =============================================================================
# Bij een significante Friedman-test zoeken we uit welke blokken van elkaar
# verschillen. We vergelijken elk paar blokken met een Wilcoxon signed-rank test
# (paarsgewijs, want dezelfde locaties).
#
# Bonferroni-correctie: bij 10 blokken zijn er C(10,2) = 45 vergelijkingen.
# De gecorrigeerde drempel wordt: α / 45

n_vergelijkingen = n_blokken * (n_blokken - 1) // 2
alpha_gecorrigeerd = DREMPEL_ALPHA / n_vergelijkingen

print(f"\n--- POST-HOC WILCOXON (Bonferroni-gecorrigeerd, α = {alpha_gecorrigeerd:.5f}) ---")
print(f"Aantal vergelijkingen: {n_vergelijkingen}")

resultaten = []
for (i, k1), (j, k2) in combinations(enumerate(blok_kolommen), 2):
    stat, p_val = wilcoxon(data_compleet[k1], data_compleet[k2])
    significant = p_val < alpha_gecorrigeerd
    mediaan_verschil = np.median(data_compleet[k2] - data_compleet[k1])
    resultaten.append({
        'blok_1': blok_labels[i],
        'blok_2': blok_labels[j],
        'W': stat,
        'p_waarde': p_val,
        'p_gecorrigeerd': p_val * n_vergelijkingen,   # Bonferroni-gecorrigeerde p
        'significant': significant,
        'mediaan_verschil_dBA': round(mediaan_verschil, 2)
    })

df_resultaten = pd.DataFrame(resultaten)
sig_resultaten = df_resultaten[df_resultaten['significant']]
print(f"\nSignificante paren: {len(sig_resultaten)} / {n_vergelijkingen}")
print(df_resultaten[['blok_1', 'blok_2', 'p_gecorrigeerd', 'significant', 'mediaan_verschil_dBA']].to_string(index=False))

# =============================================================================
# STAP 4 — Grafiek 1: Gecombineerde boxplot met mediaanwaarden
# =============================================================================

laad_font(FONT_PAD)
fig, ax = plt.subplots(figsize=(14, 6))
fig.patch.set_facecolor(ACHTERGROND_KLEUR)
ax.set_facecolor(ACHTERGROND_KLEUR)

kleur_lijst = [KLEUREN[k.replace("G_", "")] for k in blok_kolommen]

bp = ax.boxplot(
    [data_compleet[k].values for k in blok_kolommen],
    patch_artist=True,
    medianprops=dict(color='black', linewidth=2),
    whiskerprops=dict(linewidth=1.2),
    capprops=dict(linewidth=1.2),
    flierprops=dict(marker='o', markersize=4, alpha=0.5)
)

for patch, kleur in zip(bp['boxes'], kleur_lijst):
    patch.set_facecolor(kleur)
    patch.set_alpha(0.85)

for flier, kleur in zip(bp['fliers'], kleur_lijst):
    flier.set_markerfacecolor(kleur)

# Mediaanwaarde rechts van de mediaanlijn
for i, k in enumerate(blok_kolommen):
    mediaan = data_compleet[k].median()
    ax.text(i + 1 + 0.27, mediaan, f"{mediaan:.1f}",
            ha='left', va='center', fontsize=18, fontweight='bold')

legenda_elementen = [
    mpatches.Patch(facecolor="#fbd4b4", label='29 augustus'),
    mpatches.Patch(facecolor="#79d1ff", label='30 augustus'),
    mpatches.Patch(facecolor="#92d050", label='31 augustus'),
]
ax.legend(handles=legenda_elementen, fontsize=16, loc='upper left')

# Verticale scheidingslijnen tussen de dagen
for x in [2.5, 6.5]:
    ax.axvline(x=x, color='gray', linestyle='--', linewidth=1, alpha=0.6)

ax.set_xticks(range(1, n_blokken + 1))
ax.set_xticklabels(blok_labels, rotation=30, ha='right', fontsize=16)
ax.set_ylabel("Gemiddeld geluidsniveau (dBA)", fontsize=18)
ax.tick_params(axis='y', labelsize=16)
ax.set_title(
    f"Medianen van gemiddelde geluidsniveaus per festivalblok",
    fontsize=25, fontweight='bold'
)
ax.set_xlim(0.5, n_blokken + 1.2)
ax.grid(axis='y', linestyle=':', alpha=0.5)
plt.tight_layout()
plt.savefig('C:/Users/Gebruiker/Documents/DATA/Grafieken/Exploratief/Boxplot_blokken.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Grafiek 1 opgeslagen: Boxplot_blokken.png")

# =============================================================================
# STAP 5 — Grafiek 2: Heatmap van post-hoc p-waarden
# =============================================================================
laad_font(FONT_PAD)
p_matrix = pd.DataFrame(np.nan, index=blok_labels, columns=blok_labels)
for _, rij in df_resultaten.iterrows():
    p_matrix.loc[rij['blok_1'], rij['blok_2']] = rij['p_gecorrigeerd']
    p_matrix.loc[rij['blok_2'], rij['blok_1']] = rij['p_gecorrigeerd']

fig, ax = plt.subplots(figsize=(11, 9))
fig.patch.set_facecolor(ACHTERGROND_KLEUR)
ax.set_facecolor(ACHTERGROND_KLEUR)

log_p = np.log10(p_matrix.values.astype(float))
log_p_masked = np.ma.masked_invalid(log_p)

# Wit = lage p-waarde (significant), groen = hoge p-waarde (niet significant)
from matplotlib.colors import LinearSegmentedColormap
cmap_wg = LinearSegmentedColormap.from_list("wit_blauw", ["white", "#1e64c8"])
im = ax.imshow(log_p_masked, cmap=cmap_wg, vmin=-4, vmax=0, aspect='equal')

ax.set_xticks(range(n_blokken))
ax.set_yticks(range(n_blokken))
ax.set_xticklabels(blok_labels, rotation=45, ha='right', fontsize=12)
ax.set_yticklabels(blok_labels, fontsize=15)

# Celinhoud: p-waarde als tekst
for i in range(n_blokken):
    for j in range(n_blokken):
        val = p_matrix.values[i, j]
        if not np.isnan(val):
            tekst = f"{val:.3f}" if val >= 0.001 else "<0.001"
            kleur_tekst = 'black' if val > 0.01 else 'black'
            ax.text(j, i, tekst, ha='center', va='center', fontsize=12, color='black')

# Dunne hairline rasters tussen cellen
for i in range(n_blokken + 1):
    ax.axhline(i - 0.5, color='black', linewidth=0.4, zorder=3)
    # Verticale lijnen enkel tekenen binnen het datagebied (vanaf rij 0, niet in labelrij)
    ax.axvline(i - 0.5, color='black', linewidth=0.4, zorder=3,
               ymin=0, ymax=(n_blokken - 0.5) / (n_blokken + 1.7))

# Randen rond niet-significante paren
niet_sig = df_resultaten[~df_resultaten['significant']]
for _, rij in niet_sig.iterrows():
    xi = blok_labels.index(rij['blok_1'])
    yi = blok_labels.index(rij['blok_2'])
    for (r, c) in [(xi, yi), (yi, xi)]:
        ax.add_patch(mpatches.Rectangle(
            (c - 0.5, r - 0.5), 1, 1,
            fill=False, edgecolor=SIGNIFICANTIE_LIJN, linewidth=2, zorder=5
        ))

# Dag-scheiding: dikke grijze lijn na blok 2 (29 aug) en blok 6 (30 aug)
for grens in [1.5, 5.5]:
    ax.axhline(grens, color='#555555', linewidth=2.5, zorder=4)
    ax.axvline(grens, color='#555555', linewidth=2.5, zorder=4)

# Dag-labels boven de kolommen
for x_mid, label in [(0.5, '29 aug'), (3.5, '30 aug'), (7.5, '31 aug')]:
    ax.text(x_mid, -1.5, label, ha='center', va='center',
            fontsize=18, fontweight='bold', transform=ax.transData)

ax.set_xlim(-0.5, n_blokken - 0.5)
ax.set_ylim(n_blokken - 0.5, -2.2)   # extra ruimte boven voor dag-labels

cbar = plt.colorbar(im, ax=ax, shrink=0.7)
cbar.set_label("log₁₀(p-waarde, Bonferroni-gecorrigeerd)", fontsize=15)
cbar.set_ticks([-4, -3, -2, -1, 0])
cbar.set_ticklabels(['0.0001', '0.001', '0.01', '0.1', '1.0'], fontsize=12)

ax.set_title(
    f"Post-hoc Wilcoxon p-waarden (Bonferroni, α={alpha_gecorrigeerd:.4f})\nRode rand = niet-significant",
    fontsize=20, fontweight='bold'
)
plt.tight_layout()
plt.savefig('C:/Users/Gebruiker/Documents/DATA/Grafieken/Exploratief/Heatmap_posthoc.png', dpi=150, bbox_inches='tight')
plt.close()
print("Grafiek 2 opgeslagen: grafiek2_heatmap_posthoc.png")

print("\nKlaar!")