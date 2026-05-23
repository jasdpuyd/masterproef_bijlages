import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
from datetime import datetime, timedelta
from pathlib import Path

# ------------------------------------------------------------------ #
#  1  PADEN 
# ------------------------------------------------------------------ #

INVOERMAP  = r"C:\Users\Gebruiker\Documents\DATA\Metingen\Interpolatietabellen"
UITVOERMAP = r"C:\Users\Gebruiker\Documents\DATA\Grafieken\Verlopen"
# Lettertype
FONT_PAD   = r"C:\Users\Gebruiker\AppData\Local\Microsoft\Windows\Fonts\UGentPannoText-Medium.ttf"
# ------------------------------------------------------------------ #
#  2  VASTE ASSEN
# ------------------------------------------------------------------ #

X_START = "15:45:00"
X_EIND  = "00:15:00"

# Standaard y-as (Audiomoth + H-toestellen)
Y_MIN_STD = 30
Y_MAX_STD = 85

# Y-as voor FOH-metingen
Y_MIN_FOH = 40
Y_MAX_FOH = 110

# Glijdend gemiddelde venster in seconden (1 meting/sec verondersteld)
ROLLING_SECONDEN = 5

# ------------------------------------------------------------------ #
#  3  OVERIGE INSTELLINGEN
# ------------------------------------------------------------------ #

DPI           = 150
FIGUURGROOTTE = (16, 5)
LIJNKLEUR     = "#1e64c8"
ACHTERGROND   = "#ffffff"
RASTERKLEUR   = "#dddddd"

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
# ------------------------------------------------------------------ #
#  4  CSV INLEZEN
# ------------------------------------------------------------------ #

def parse_dba(kolom):
    if pd.api.types.is_float_dtype(kolom):
        return kolom
    kolom = (
        kolom.astype(str)
        .str.strip()
        .str.strip('"')
        .str.lstrip("0")
        .str.replace(",", ".", regex=False)
    )
    kolom = kolom.replace("", "0")
    return pd.to_numeric(kolom, errors="coerce")


def parse_datetime(datum, tijd):
    gecombineerd = datum.astype(str).str.strip() + " " + tijd.astype(str).str.strip()
    for fmt in ["%d-%m-%Y %H:%M:%S", "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M", "%d-%m-%Y %H:%M"]:
        try:
            resultaat = pd.to_datetime(gecombineerd, format=fmt)
            if resultaat.notna().sum() > 0:
                return resultaat
        except Exception:
            continue
    return pd.to_datetime(gecombineerd, dayfirst=True, errors="coerce")


def is_foh(df):
    """Geeft True als dit een FOH-meting is (Toestel-kolom bevat 'FOH')."""
    if "Toestel" in df.columns:
        return df["Toestel"].astype(str).str.upper().str.contains("FOH").any()
    return False


def laad_csv(pad):
    df = None
    for encoding in ("utf-8-sig", "latin-1", "cp1252"):
        try:
            df = pd.read_csv(pad, sep=",", encoding=encoding, dtype=str)
            if {"Datum", "Tijd", "dBA"}.issubset(df.columns):
                break
            df = None
        except Exception:
            df = None
            continue

    if df is None:
        print("  ! Kon niet inlezen of verplichte kolommen ontbreken: {}".format(pad))
        return None

    df["dBA"]      = parse_dba(df["dBA"])
    df["Datetime"] = parse_datetime(df["Datum"], df["Tijd"])
    df = df.dropna(subset=["Datetime", "dBA"]).sort_values("Datetime").reset_index(drop=True)

    if df.empty:
        print("  ! Geen geldige rijen na verwerking: {}".format(pad))
        return None

    # Glijdend gemiddelde op basis van tijd
    # Stel Datetime in als index voor time-based rolling
    df = df.set_index("Datetime")
    venster = "{}s".format(ROLLING_SECONDEN)
    df["dBA_smooth"] = df["dBA"].rolling(venster, min_periods=1).mean()
    df = df.reset_index()

    return df

# ------------------------------------------------------------------ #
#  5  GRAFIEK MAKEN
# ------------------------------------------------------------------ #

def maak_grafiek(df, csv_pad, uitvoermap):
    bestandsnaam = Path(csv_pad).stem
    submap_naam  = Path(csv_pad).parent.name

    toestel_raw = str(df["Toestel"].iloc[0]) if "Toestel" in df.columns else ""
    if toestel_raw.upper().startswith("H"):
        toestel = "Extech SDL600"
    elif toestel_raw.upper().startswith("A"):
        toestel = "AudioMoth"
    else:
        toestel = "FOH"
    meetpunt = str(df["Meetpunt"].iloc[0]) if "Meetpunt" in df.columns else "-"
    datum_str = df["Datetime"].iloc[0].strftime("%d/%m/%Y")

    # Y-as grenzen afhankelijk van toesteltype
    if is_foh(df):
        y_min, y_max = Y_MIN_FOH, Y_MAX_FOH
    else:
        y_min, y_max = Y_MIN_STD, Y_MAX_STD

    # X-as grenzen
    meetdatum    = df["Datetime"].iloc[0].date()
    volgende_dag = meetdatum + timedelta(days=1)
    x_start = datetime.combine(meetdatum,    datetime.strptime(X_START, "%H:%M:%S").time())
    x_eind  = datetime.combine(volgende_dag, datetime.strptime(X_EIND,  "%H:%M:%S").time())

    fig, ax = plt.subplots(figsize=FIGUURGROOTTE)
    fig.patch.set_facecolor(ACHTERGROND)
    ax.set_facecolor(ACHTERGROND)

    # Ruwe data zeer licht op de achtergrond
    ax.plot(df["Datetime"], df["dBA"],
            color=LIJNKLEUR, linewidth=0.4, alpha=0.25)

    # Glijdend gemiddelde als hoofdlijn
    ax.plot(df["Datetime"], df["dBA_smooth"],
            color=LIJNKLEUR, linewidth=1.1, alpha=0.95,
            label="{} s gemiddelde".format(ROLLING_SECONDEN))

    ax.fill_between(df["Datetime"], df["dBA_smooth"], y_min,
                    alpha=0.08, color=LIJNKLEUR)

    ax.set_xlim(x_start, x_eind)
    ax.set_ylim(y_min, y_max)

    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=15))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.xticks(rotation=45, ha="right", fontsize=18)

    ax.set_yticks(range(y_min, y_max + 1, 10))
    plt.yticks(fontsize=18)

    ax.set_xlabel("Tijd", fontsize=24)
    ax.set_ylabel("Geluidsniveau (dBA)", fontsize=24)
    ax.set_title(
        "Geluidsverloop - {}  |  Toestel: {}  |  Meetpunt: {}".format(
            datum_str, toestel, meetpunt),
        fontsize=30, fontweight="bold", pad=12
    )

    stats = "Gem: {:.1f} dBA\nMax: {:.1f} dBA\nMin: {:.1f} dBA".format(
        df["dBA"].mean(), df["dBA"].max(), df["dBA"].min()
    )
    ax.text(0.99, 0.97, stats,
            transform=ax.transAxes, fontsize=20,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      alpha=0.75, edgecolor="#cccccc"))

    ax.grid(True, which="major", color=RASTERKLEUR, linewidth=0.6, linestyle="--")
    ax.grid(True, which="minor", color=RASTERKLEUR, linewidth=0.3, linestyle=":")
    ax.set_axisbelow(True)

    dag_uitvoermap = os.path.join(uitvoermap, submap_naam)
    os.makedirs(dag_uitvoermap, exist_ok=True)
    uitvoerpad = os.path.join(dag_uitvoermap, bestandsnaam + ".png")
    plt.tight_layout()
    plt.savefig(uitvoerpad, dpi=DPI, bbox_inches="tight")
    plt.close()
    print("  OK  {}".format(uitvoerpad))

# ------------------------------------------------------------------ #
#  6  HOOFDPROGRAMMA
# ------------------------------------------------------------------ #

def verwerk_alle_csv(invoermap, uitvoermap):
    laad_font(FONT_PAD)
    os.makedirs(uitvoermap, exist_ok=True)

    csv_bestanden = sorted(glob.glob(
        os.path.join(invoermap, "**", "*.csv"), recursive=True
    ))

    if not csv_bestanden:
        print("\nGeen CSV-bestanden gevonden in: {}".format(invoermap))
        print("Controleer of het pad correct is.")
        return

    print("\n{} CSV-bestand(en) gevonden.".format(len(csv_bestanden)))
    print("Output map: {}\n".format(uitvoermap))

    geslaagd, mislukt = 0, 0
    for pad in csv_bestanden:
        print("-> {}".format(pad))
        df = laad_csv(pad)
        if df is None:
            mislukt += 1
            continue
        maak_grafiek(df, pad, uitvoermap)
        geslaagd += 1

    print("\n" + "-" * 55)
    print("Geslaagd : {}".format(geslaagd))
    if mislukt:
        print("Mislukt  : {}".format(mislukt))
    print("-" * 55 + "\n")


if __name__ == "__main__":
    verwerk_alle_csv(INVOERMAP, UITVOERMAP)