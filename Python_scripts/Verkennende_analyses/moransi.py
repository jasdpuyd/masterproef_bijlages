import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
import geopandas as gpd
import libpysal
import esda
from esda.moran import Moran_Local
import contextily as ctx
import warnings
warnings.filterwarnings("ignore")

# ── CONFIGURATIE ──────────────────────────────────────────────────────────────
SHAPEFILE       = r"C:\Users\Gebruiker\Documents\DATA\GIS_analyse\912_centroides.shp"   
STANCE_KOLOM    = "Stance"               
NEGATIEF_WAARDE = "Negatief"             
K_BUREN         = 5                      
PERMUTATIES     = 9999                   
P_DREMPEL       = 0.05                   
OUTPUT_MAP      = r"C:\Users\Gebruiker\Documents\DATA\Grafieken\MoransI"

# Lettertype
FONT_PAD        = r"C:\Users\Gebruiker\AppData\Local\Microsoft\Windows\Fonts\UGentPannoText-Medium.ttf"
# ─────────────────────────────────────────────────────────────────────────────

# ── KLEURENSCHEMA (zelfde als correlatie.py) ──────────────────────────────────
KLEUR_NEG   = "#C0392B"
KLEUR_POS   = "#2980B9"
KLEUR_GEM   = "#79d1ff"
KLEUR_MAX   = "#1e64c8"
ACHTERGROND = "#F7F9FB"
RASTER      = "#DDE3EA"

# LISA-specifieke kleuren
KLEUR_HH = "#C0392B"   # negatief cluster
KLEUR_LL = "#2980B9"   # niet-negatief cluster
KLEUR_HL = "#E67E22"   # hoog omringd door laag
KLEUR_LH = "#79d1ff"   # laag omringd door hoog
KLEUR_NS = "#BDC3C7"   # niet significant


# ── HULPFUNCTIES ──────────────────────────────────────────────────────────────

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


def stel_stijl_in():
    """Zet globale matplotlib-stijl."""
    plt.rcParams.update({
        "figure.facecolor":  ACHTERGROND,
        "axes.facecolor":    ACHTERGROND,
        "axes.edgecolor":    "#BDC3C7",
        "axes.grid":         True,
        "grid.color":        RASTER,
        "grid.linewidth":    0.8,
        "font.size":         11,
        "axes.titlesize":    13,
        "axes.titleweight":  "bold",
        "axes.labelsize":    11,
        "xtick.labelsize":   10,
        "ytick.labelsize":   10,
        "legend.framealpha": 0.9,
        "legend.edgecolor":  "#BDC3C7",
    })


def uitvoerpad(bestandsnaam):
    """Koppel OUTPUT_MAP aan bestandsnaam."""
    import os
    if OUTPUT_MAP:
        os.makedirs(OUTPUT_MAP, exist_ok=True)
        return os.path.join(OUTPUT_MAP, bestandsnaam)
    return bestandsnaam


# ── DATA & BEREKENINGEN ───────────────────────────────────────────────────────

def laad_data(shapefile, stance_kolom, negatief_waarde):
    gdf = gpd.read_file(shapefile)
    if stance_kolom not in gdf.columns:
        raise ValueError(
            f"Kolom '{stance_kolom}' niet gevonden. "
            f"Beschikbare kolommen: {gdf.columns.tolist()}"
        )
    gdf["binary"] = (gdf[stance_kolom] == negatief_waarde).astype(int)
    n_neg    = int(gdf["binary"].sum())
    n_nonneg = len(gdf) - n_neg
    print(f"Data geladen: {len(gdf)} punten  |  "
          f"Negatief: {n_neg}  |  Niet-negatief: {n_nonneg}")
    return gdf


def maak_gewichtenmatrix(gdf, k):
    coords = np.column_stack([gdf.geometry.x, gdf.geometry.y])
    w = libpysal.weights.KNN(coords, k=k)
    w.transform = "r"
    print(f"Gewichtenmatrix: k={k} buren, "
          f"gemiddeld {w.mean_neighbors:.1f} buren per punt")
    return w

def bereken_moran(gdf, w, permutaties, p_drempel):
    mi   = esda.Moran(gdf["binary"], w, permutations=permutaties)
    lisa = Moran_Local(gdf["binary"], w, permutations=permutaties)

    quad_map = {1: "HH", 2: "LH", 3: "LL", 4: "HL"}
    sig    = lisa.p_sim < p_drempel
    labels = np.full(len(gdf), "NS", dtype=object)
    for i in range(len(gdf)):
        if sig[i]:
            labels[i] = quad_map[lisa.q[i]]
    gdf["LISA"] = labels

    print(f"\n=== GLOBALE MORAN'S I ===")
    print(f"  I-waarde    : {mi.I:.4f}")
    print(f"  Verwachte I : {mi.EI:.4f}")
    print(f"  Z-score     : {mi.z_sim:.4f}")
    print(f"  p-waarde    : {mi.p_sim:.4f}")
    print(f"  Significant : {'Ja (p < 0.05)' if mi.p_sim < 0.05 else 'Nee'}")
    print(f"\n=== LISA CLUSTERS (p < {p_drempel}) ===")
    print(gdf["LISA"].value_counts().to_string())
    return mi, lisa


# ── FIGUREN ───────────────────────────────────────────────────────────────────

def fig1_stances(gdf):
    """Figuur 1 — Ruimtelijke verdeling van stances."""
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.suptitle("Ruimtelijke verdeling stances",
                 fontsize=14, fontweight="bold")

    kleuren = gdf[STANCE_KOLOM].map(
        {NEGATIEF_WAARDE: KLEUR_NEG}
    ).fillna(KLEUR_POS)

    ax.scatter(gdf.geometry.x, gdf.geometry.y,
               c=kleuren, s=70, edgecolors="#7F8C8D",
               linewidths=0.5, zorder=3)

    n_neg    = int(gdf["binary"].sum())
    n_nonneg = len(gdf) - n_neg
    handles = [
        mpatches.Patch(color=KLEUR_NEG, label=f"Negatief (n={n_neg})"),
        mpatches.Patch(color=KLEUR_POS, label=f"Niet-negatief (n={n_nonneg})"),
    ]
    ax.legend(handles=handles, fontsize=10)
    ax.set_xlabel("X-coördinaat (m)")
    ax.set_ylabel("Y-coördinaat (m)")
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9)

    plt.tight_layout()
    pad = uitvoerpad("1_stances_kaart.png")
    plt.savefig(pad, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figuur 1 opgeslagen: {pad}")


def fig2_referentieverdeling(gdf, mi, k):
    """Figuur 2 — Histogram van gesimuleerde Moran's I-waarden."""
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.suptitle(
        f"Globale Moran's I — referentieverdeling (k={k}, n={len(gdf)})",
        fontsize=20, fontweight="bold"
    )

    ax.hist(mi.sim, bins=50, color=KLEUR_GEM, edgecolor=ACHTERGROND,
            alpha=0.85, label="Gesimuleerde I-waarden (permutaties)", zorder=3)
    ax.axvline(mi.I,  color=KLEUR_NEG, linewidth=2,
               label=f"Moran's I = {mi.I:.4f}", zorder=3)
    ax.axvline(mi.EI, color="#7F8C8D", linewidth=1.5, linestyle="--",
               label=f"Verwachte I = {mi.EI:.4f}", zorder=3)

    # zorder=5 zodat het tekstvak boven de verticale lijnen valt
    sig_tekst = (
        f"p = {mi.p_sim:.4f}  —  "
        + ("significant (p < 0.05)" if mi.p_sim < 0.05 else "niet significant")
    )
    ax.text(0.97, 0.96, sig_tekst, transform=ax.transAxes,
            ha="right", va="top", fontsize=15, zorder=5,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#BDC3C7", alpha=1.0))

    ax.set_xlabel("Gesimuleerde I-waarden", fontsize=15)
    ax.set_ylabel("Frequentie", fontsize=15)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=15)
    ax.legend(fontsize=15)

    plt.tight_layout()
    pad = uitvoerpad("2_referentieverdeling.png")
    plt.savefig(pad, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figuur 2 opgeslagen: {pad}")


def fig3_scatterplot(gdf, mi, w):
    """Figuur 3 — Moran scatterplot met kwadranten."""
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.suptitle("Moran scatterplot", fontsize=14, fontweight="bold")

    y       = gdf["binary"]
    y_std   = (y - y.mean()) / y.std()
    lag     = libpysal.weights.lag_spatial(w, y)
    lag_std = (lag - lag.mean()) / lag.std()

    kleur_map = {
        "HH": KLEUR_HH, "LL": KLEUR_LL,
        "HL": KLEUR_HL, "LH": KLEUR_LH, "NS": KLEUR_NS,
    }
    scatter_kleuren = [kleur_map.get(lbl, KLEUR_NS) for lbl in gdf["LISA"]]

    ax.scatter(y_std, lag_std, c=scatter_kleuren, s=60,
               edgecolors="#7F8C8D", linewidths=0.4, zorder=3, alpha=0.9)

    m_coef, b_coef = np.polyfit(y_std, lag_std, 1)
    x_line = np.linspace(y_std.min(), y_std.max(), 200)
    ax.plot(x_line, m_coef * x_line + b_coef, color=KLEUR_MAX, linewidth=2,
            zorder=2, label=f"Regressielijn (helling = {m_coef:.4f})")

    ax.axhline(0, color="#BDC3C7", linewidth=1.0, linestyle="--", zorder=1)
    ax.axvline(0, color="#BDC3C7", linewidth=1.0, linestyle="--", zorder=1)

    for tekst, xp, yp, ha, va in [
        ("HH", 0.97, 0.97, "right", "top"),
        ("LH", 0.03, 0.97, "left",  "top"),
        ("LL", 0.03, 0.03, "left",  "bottom"),
        ("HL", 0.97, 0.03, "right", "bottom"),
    ]:
        ax.text(xp, yp, tekst, transform=ax.transAxes,
                color="#BDC3C7", fontsize=16, fontweight="bold",
                ha=ha, va=va, alpha=0.7)

    ax.set_xlabel("Gestandaardiseerde stance")
    ax.set_ylabel("Ruimtelijke lag (gemiddelde buurwaarden)")
    ax.set_axisbelow(True)
    ax.legend(fontsize=10)

    plt.tight_layout()
    pad = uitvoerpad("3_moran_scatterplot.png")
    plt.savefig(pad, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figuur 3 opgeslagen: {pad}")


# ── HOOFDPROGRAMMA ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    laad_font(FONT_PAD)
    stel_stijl_in()

    gdf      = laad_data(SHAPEFILE, STANCE_KOLOM, NEGATIEF_WAARDE)
    w        = maak_gewichtenmatrix(gdf, K_BUREN)
    mi, lisa = bereken_moran(gdf, w, PERMUTATIES, P_DREMPEL)

    fig1_stances(gdf)
    fig2_referentieverdeling(gdf, mi, K_BUREN)
    fig3_scatterplot(gdf, mi, w)

    print("\nKlaar. Vier figuren opgeslagen.")