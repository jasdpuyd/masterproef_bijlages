import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# KLEURINSTELLINGEN
# =============================================================================

KLEUR_29 = "#fbd4b4"        # Kleur voor 29 augustus
KLEUR_30 = "#79d1ff"        # Kleur voor 30 augustus
KLEUR_31 = "#92d050"        # Kleur voor 31 augustus
ACHTERGROND_KLEUR = "#FFFFFF"
SIGNIFICANTIE_LIJN = "#CC0000"
DREMPEL_ALPHA = 0.05
# Lettertype
FONT_PAD   = r"C:\Users\Gebruiker\AppData\Local\Microsoft\Windows\Fonts\UGentPannoText-Medium.ttf"
# =============================================================================
# STAP 1 — Data inladen
# =============================================================================

df = pd.read_csv(
    'C:/Users/Gebruiker/Documents/DATA/Resultaten/DATASET.csv',
    sep=';',
    encoding='utf-8-sig',
    decimal=','
)

blokken_per_dag = {
    '29 aug': ['G_29aug_blok1', 'G_29aug_blok2'],
    '30 aug': ['G_30aug_blok1', 'G_30aug_blok2', 'G_30aug_blok3', 'G_30aug_blok4'],
    '31 aug': ['G_31aug_blok1', 'G_31aug_blok2', 'G_31aug_blok3', 'G_31aug_blok4'],
}

alle_kolommen = [k for kolommen in blokken_per_dag.values() for k in kolommen]
data = df[alle_kolommen].apply(pd.to_numeric, errors='coerce').dropna()
print(f"Locaties met volledige data: {len(data)} / {len(df)}")

# =============================================================================
# STAP 2 — Energetisch gemiddelde per dag per locatie
# =============================================================================

def energetisch_gemiddelde(rij, kolommen):
    waarden = rij[kolommen].values.astype(float)
    return 10 * np.log10(np.mean(10 ** (waarden / 10)))

for dag, kolommen in blokken_per_dag.items():
    kolomnaam = f"L_{dag.replace(' ', '_')}"
    data[kolomnaam] = data.apply(lambda rij: energetisch_gemiddelde(rij, kolommen), axis=1)

print("\nEnergetisch gemiddelde per dag (eerste 5 locaties):")
print(data[['L_29_aug', 'L_30_aug', 'L_31_aug']].head())

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
# STAP 3 — Wilcoxon signed-rank toets per dagpaar
# =============================================================================

n_vergelijkingen = 3
alpha_gecorrigeerd = DREMPEL_ALPHA / n_vergelijkingen

dag_paren = [
    ('29 aug', '30 aug'),
    ('29 aug', '31 aug'),
    ('30 aug', '31 aug'),
]

dag_kolommen = {
    '29 aug': 'L_29_aug',
    '30 aug': 'L_30_aug',
    '31 aug': 'L_31_aug',
}

print(f"\n--- WILCOXON SIGNED-RANK TOETS PER DAGPAAR ---")
print(f"Bonferroni-gecorrigeerde drempel: α = {alpha_gecorrigeerd:.4f}\n")

dag_resultaten = []
for dag1, dag2 in dag_paren:
    k1 = dag_kolommen[dag1]
    k2 = dag_kolommen[dag2]
    stat, p_val = wilcoxon(data[k1], data[k2])
    p_gecorrigeerd = p_val * n_vergelijkingen
    mediaan_verschil = np.median(data[k2] - data[k1])
    significant = p_val < alpha_gecorrigeerd

    dag_resultaten.append({
        'dag_1': dag1,
        'dag_2': dag2,
        'W': stat,
        'p_waarde': p_val,
        'p_gecorrigeerd': round(p_gecorrigeerd, 6),
        'significant': significant,
        'mediaan_verschil_dBA': round(mediaan_verschil, 2)
    })

    print(f"{dag1} vs. {dag2}:")
    print(f"  W = {stat:.1f},  p = {p_val:.6f},  p (Bonferroni) = {p_gecorrigeerd:.6f}")
    print(f"  Mediaan verschil: {mediaan_verschil:+.2f} dBA")
    print(f"  → {'SIGNIFICANT' if significant else 'Niet significant'}\n")

df_dag_resultaten = pd.DataFrame(dag_resultaten)

# =============================================================================
# STAP 4 — Grafiek 1: Boxplot per meetdag
# =============================================================================

laad_font(FONT_PAD)
fig, ax = plt.subplots(figsize=(9, 6))
fig.patch.set_facecolor(ACHTERGROND_KLEUR)
ax.set_facecolor(ACHTERGROND_KLEUR)

dag_labels = ['29 aug', '30 aug', '31 aug']
dag_kleuren = [KLEUR_29, KLEUR_30, KLEUR_31]
dag_data = [data[dag_kolommen[d]].values for d in dag_labels]

bp = ax.boxplot(
    dag_data,
    patch_artist=True,
    medianprops=dict(color='black', linewidth=2),
    whiskerprops=dict(linewidth=1.2),
    capprops=dict(linewidth=1.2),
    flierprops=dict(marker='o', markersize=5, alpha=0.5)
)

for patch, kleur in zip(bp['boxes'], dag_kleuren):
    patch.set_facecolor(kleur)
    patch.set_alpha(0.85)

legenda_elementen = [
    mpatches.Patch(facecolor="#fbd4b4", label='29 augustus'),
    mpatches.Patch(facecolor="#79d1ff", label='30 augustus'),
    mpatches.Patch(facecolor="#92d050", label='31 augustus'),
]
ax.legend(handles=legenda_elementen, fontsize=16, loc='upper right')

# Mediaanwaarde rechts van de mediaanlijn
for i, d in enumerate(dag_labels):
    mediaan = np.median(dag_data[i])
    ax.text(i + 1 + 0.2, mediaan, f"{mediaan:.1f}",
            ha='left', va='center', fontsize=18, fontweight='bold')

ax.set_xlim(0.5, 3.9)

for flier, kleur in zip(bp['fliers'], dag_kleuren):
    flier.set_markerfacecolor(kleur)

ax.set_xticks([1, 2, 3])
ax.set_xticklabels(dag_labels, fontsize=16)
ax.set_ylabel("Gemiddeld geluidsniveau (dBA)", fontsize=18)
ax.tick_params(axis='y', labelsize=16)
ax.set_title("Medianen van gemiddelde geluidsniveaus per dag", fontsize=23, fontweight='bold')

# Significantie-annotaties boven de boxplot
y_max = max([d.max() for d in dag_data])
y_stap = (ax.get_ylim()[1] - y_max) * 0.25

y1 = y_max + y_stap
y2 = y1 + y_stap * 1.4
y3 = y2 + y_stap * 1.4

for (_, rij), y in zip(df_dag_resultaten.iterrows(), [y1, y2, y3]):
    x1 = dag_labels.index(rij['dag_1']) + 1
    x2 = dag_labels.index(rij['dag_2']) + 1

ax.set_ylim(bottom=min([d.min() for d in dag_data]) - 3,
            top=y3 + y_stap * 2)

ax.grid(axis='y', linestyle=':', alpha=0.5)
plt.tight_layout()
plt.savefig('C:/Users/Gebruiker/Documents/DATA/Grafieken/Exploratief_dagen/Boxplot_dagen.png', dpi=150, bbox_inches='tight')
plt.close()
print("Grafiek 1 opgeslagen: grafiek_boxplot_dagen.png")

# Sla resultaten op als CSV
df_dag_resultaten.to_csv(r'C:/Users/Gebruiker/Documents/DATA/Grafieken/Exploratief_dagen/resultaat.csv', index=False, sep=';', decimal=',')

print("\nKlaar!")