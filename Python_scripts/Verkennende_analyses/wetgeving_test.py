"""
Festival Geluidsoverschrijding Analyse
=======================================
Controleert per pixel of het energetisch gemiddelde (Leq) over een
voortschrijdend venster van 15 minuten (900 seconden) de wettelijke
drempel van 85 dBA overschrijdt.

Verwachte mapstructuur:
    Raster_2/
        30_augustus/
            1sec/
                20250830_161111.tif
                20250830_161112.tif
                ...
            (andere submappen worden genegeerd)
        31_augustus/
            1sec/
                20250831_...tif
                ...

Output:
    output/
        overschrijdingen_count.tif       - per pixel: aantal vensters in overtreding
        overschrijdingen_tijdstippen.csv - per venster: tijdstip + aantal pixels
        overschrijdingen_locaties.csv    - per pixel (x,y): aantal + eerste/laatste overtreding
        samenvatting.txt                 - leesbare samenvatting
"""

import os
import sys
import numpy as np
import rasterio
from rasterio.transform import xy
from datetime import datetime, timedelta
from pathlib import Path
import csv
import time

# ─── CONFIGURATIE ────────────────────────────────────────────────────────────

DATA_DIR   = Path("C:\\Users\\Gebruiker\\Documents\\DATA\\Raster_2")     
OUTPUT_DIR = Path("C:\\Users\\Gebruiker\\Documents\\DATA\\Wetgeving")
DAGMAPPEN  = ["30_augustus", "31_augustus"]

VENSTER_SECONDEN = 900             # 15 minuten
DREMPEL_DBA      = 85.0            # wettelijke grens
CHUNK_SIZE       = 1000            # aantal rasters per chunk in geheugen

# ─── HULPFUNCTIES ─────────────────────────────────────────────────────────────

def parse_tijdstip(bestandsnaam: str) -> datetime:
    """Extraheer tijdstip uit bestandsnaam YYYYMMDD_HHMMSS.tif"""
    stam = Path(bestandsnaam).stem   # '20250830_161111'
    return datetime.strptime(stam, "%Y%m%d_%H%M%S")


def verzamel_bestanden(data_dir: Path, dagmappen: list) -> list[tuple[datetime, Path]]:
    """Geeft gesorteerde lijst van (tijdstip, pad) voor alle rasters."""
    bestanden = []
    for dag in dagmappen:
        sec_map = data_dir / dag / "1sec"
        if not sec_map.exists():
            print(f"  [!] Map niet gevonden: {sec_map} — wordt overgeslagen")
            continue
        for pad in sec_map.glob("*.tif"):
            try:
                ts = parse_tijdstip(pad.name)
                bestanden.append((ts, pad))
            except ValueError:
                print(f"  [!] Ongeldige bestandsnaam: {pad.name} — wordt overgeslagen")
    bestanden.sort(key=lambda x: x[0])
    return bestanden


def lees_raster_meta(pad: Path):
    """Lees metadata (shape, transform, crs) uit één raster."""
    with rasterio.open(pad) as src:
        return src.profile.copy(), src.shape, src.transform, src.crs


def laad_chunk(bestanden_chunk: list[tuple[datetime, Path]], shape: tuple) -> np.ndarray:
    """
    Laad een lijst rasters als 3D float32 array (tijd, rijen, kolommen).
    Ontbrekende bestanden worden opgevuld met NaN.
    """
    n = len(bestanden_chunk)
    rijen, kolommen = shape
    blok = np.empty((n, rijen, kolommen), dtype=np.float32)
    for i, (_, pad) in enumerate(bestanden_chunk):
        with rasterio.open(pad) as src:
            blok[i] = src.read(1)
    return blok


def dba_naar_lineair(arr: np.ndarray) -> np.ndarray:
    """Zet dBA-waarden om naar lineaire energiewaarden."""
    return np.power(10.0, arr / 10.0, dtype=np.float64)


def lineair_naar_dba(arr: np.ndarray) -> np.ndarray:
    """Zet lineaire energiewaarden terug naar dBA."""
    return 10.0 * np.log10(arr)

# ─── HOOFDBEREKENING ──────────────────────────────────────────────────────────

def analyseer(data_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()

    # 1. Bestanden verzamelen
    print("Bestanden verzamelen...")
    bestanden = verzamel_bestanden(data_dir, DAGMAPPEN)
    N = len(bestanden)
    if N == 0:
        print("Geen bestanden gevonden. Controleer DATA_DIR en DAGMAPPEN.")
        sys.exit(1)
    print(f"  {N} rasters gevonden ({bestanden[0][0]} → {bestanden[-1][0]})")

    if N < VENSTER_SECONDEN:
        print(f"  [!] Minder dan {VENSTER_SECONDEN} rasters — geen volledig 15-min venster mogelijk.")
        sys.exit(1)

    # 2. Rastermetadata — scan alle rasters om maximale afmetingen te bepalen
    profiel, (RIJEN_0, KOLOMMEN_0), transform, crs = lees_raster_meta(bestanden[0][1])
    print(f"  Rastergrootte eerste raster: {RIJEN_0} rijen × {KOLOMMEN_0} kolommen")
    print(f"  Afmetingen controleren over alle rasters...", end="", flush=True)

    max_rijen = RIJEN_0
    max_kolommen = KOLOMMEN_0
    for _, pad in bestanden:
        with rasterio.open(pad) as src:
            r, k = src.shape
            if r > max_rijen:    max_rijen    = r
            if k > max_kolommen: max_kolommen = k

    RIJEN, KOLOMMEN = max_rijen, max_kolommen
    print(f" klaar.")
    print(f"  Maximale rastergrootte: {RIJEN} rijen × {KOLOMMEN} kolommen")
    print(f"  CRS: {crs}")

    # 3. Resultaatmatrices initialiseren
    #    - count_matrix: per pixel het aantal vensters in overtreding
    #    - eerste_overtreding / laatste_overtreding: tijdstipindices
    count_matrix     = np.zeros((RIJEN, KOLOMMEN), dtype=np.int32)
    eerste_idx       = np.full((RIJEN, KOLOMMEN), -1, dtype=np.int32)
    laatste_idx      = np.full((RIJEN, KOLOMMEN), -1, dtype=np.int32)
    max_leq_matrix   = np.full((RIJEN, KOLOMMEN), -np.inf, dtype=np.float32)  # hoogste Leq ooit per pixel

    # Tijdstip-log voor CSV (venster-index → tijdstip + aantal overtredende pixels)
    venster_log = []   # lijst van (tijdstip_einde_venster, n_pixels, max_leq)

    # Tijdstippenlijst voor snelle opzoek
    tijdstips = [ts for ts, _ in bestanden]

    # 4. Sliding window via circulaire buffer
    print("\nBerekening gestart...")
    print(f"  Venstergrootte: {VENSTER_SECONDEN} seconden")
    print(f"  Drempel: {DREMPEL_DBA} dBA")
    print()

    # Buffer: circulaire buffer van lineaire energiewaarden
    # Vorm: (VENSTER_SECONDEN, RIJEN, KOLOMMEN), float64
    # Pixels buiten een kleiner raster blijven 0 — dragen niet bij aan de som
    buffer      = np.zeros((VENSTER_SECONDEN, RIJEN, KOLOMMEN), dtype=np.float64)
    buffer_som  = np.zeros((RIJEN, KOLOMMEN), dtype=np.float64)  # lopende som
    buffer_ptr  = 0       # schrijfpositie in circulaire buffer
    gevuld      = 0       # hoeveel posities al ingevuld (< VENSTER_SECONDEN bij opstart)

    vorige_melding = -1

    for i in range(N):
        # Laad één raster met zijn werkelijke afmeting
        with rasterio.open(bestanden[i][1]) as src:
            raster_dba = src.read(1).astype(np.float64)
            r, k = raster_dba.shape

        # Schrijf in buffer op de werkelijke deelruimte; rest blijft 0
        nieuw_frame = np.zeros((RIJEN, KOLOMMEN), dtype=np.float64)
        nieuw_frame[:r, :k] = dba_naar_lineair(raster_dba)

        # Verwijder oude waarde uit som, voeg nieuwe toe
        buffer_som -= buffer[buffer_ptr]
        buffer[buffer_ptr] = nieuw_frame
        buffer_som += nieuw_frame
        buffer_ptr = (buffer_ptr + 1) % VENSTER_SECONDEN

        if gevuld < VENSTER_SECONDEN:
            gevuld += 1

        # Pas berekenen als buffer vol is (eerste volledig venster na index 899)
        if gevuld == VENSTER_SECONDEN:
            tijdstip_einde = tijdstips[i]

            # Leq berekenen
            leq = lineair_naar_dba(buffer_som / VENSTER_SECONDEN)

            # Overschrijdingen detecteren
            overschrijding = leq > DREMPEL_DBA           # boolean (RIJEN, KOLOMMEN)
            n_pixels = int(np.sum(overschrijding))

            # Hoogste Leq per pixel bijhouden (ook buiten overtreding)
            np.maximum(max_leq_matrix, leq, out=max_leq_matrix)

            if n_pixels > 0:
                max_leq_venster = float(np.max(leq[overschrijding]))
                venster_log.append((tijdstip_einde, n_pixels, max_leq_venster))

                # Per-pixel statistieken bijwerken
                count_matrix += overschrijding.astype(np.int32)

                # Eerste overtreding per pixel
                is_eerste = overschrijding & (eerste_idx == -1)
                eerste_idx[is_eerste] = i

                # Laatste overtreding per pixel
                laatste_idx[overschrijding] = i

        # Voortgangsindicator
        pct = int(100 * i / N)
        if pct % 5 == 0 and pct != vorige_melding:
            elapsed = time.time() - start
            eta = (elapsed / (i + 1)) * (N - i - 1) if i > 0 else 0
            print(f"  {pct:3d}%  ({i+1}/{N} rasters)  verstreken: {elapsed:.0f}s  ETA: {eta:.0f}s")
            vorige_melding = pct

    print(f"  100%  ({N}/{N} rasters)")
    print(f"\nBerekening klaar in {time.time() - start:.1f} seconden.")

    # ─── OUTPUT 1a: GeoTIFF met overschrijdingsaantal per pixel ─────────────
    tif_pad = output_dir / "overschrijdingen_count.tif"
    profiel.update(dtype=rasterio.int32, count=1, nodata=-1)
    with rasterio.open(tif_pad, "w", **profiel) as dst:
        dst.write(count_matrix.astype(np.int32), 1)
    print(f"\nGeoTIFF (telling) opgeslagen:   {tif_pad}")

    # ─── OUTPUT 1b: GeoTIFF met hoogste Leq per pixel ────────────────────────
    tif_leq_pad = output_dir / "max_leq_per_pixel.tif"
    profiel_leq = profiel.copy()
    profiel_leq.update(dtype=rasterio.float32, nodata=-9999.0)
    max_leq_output = max_leq_matrix.copy()
    max_leq_output[max_leq_output == -np.inf] = -9999.0
    with rasterio.open(tif_leq_pad, "w", **profiel_leq) as dst:
        dst.write(max_leq_output.astype(np.float32), 1)
    print(f"GeoTIFF (max Leq) opgeslagen:   {tif_leq_pad}")

    # ─── OUTPUT 2: CSV vensters ───────────────────────────────────────────────
    venster_csv = output_dir / "overschrijdingen_tijdstippen.csv"
    with open(venster_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tijdstip_begin_venster", "tijdstip_einde_venster",
                    "dag", "uur_begin", "minuut_begin",
                    "aantal_pixels_in_overtreding", "max_LAeq15min_dBA"])
        for (ts, n, max_leq) in venster_log:
            ts_begin = ts - timedelta(seconds=VENSTER_SECONDEN - 1)
            w.writerow([
                ts_begin.strftime("%Y-%m-%d %H:%M:%S"),
                ts.strftime("%Y-%m-%d %H:%M:%S"),
                ts.strftime("%Y-%m-%d"),
                ts_begin.hour, ts_begin.minute,
                n,
                round(max_leq, 2)
            ])
    print(f"Venster-CSV opgeslagen:  {venster_csv}")

    # ─── OUTPUT 3: CSV locaties ───────────────────────────────────────────────
    locaties_csv = output_dir / "overschrijdingen_locaties.csv"
    overtredende_pixels = np.argwhere(count_matrix > 0)   # (n, 2) array van (rij, kolom)
    with open(locaties_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rij", "kolom", "x_m_epsg31370", "y_m_epsg31370",
                    "aantal_vensters_overtreding", "max_LAeq15min_dBA",
                    "eerste_overtreding", "laatste_overtreding"])
        for (rij, kol) in overtredende_pixels:
            x, y = xy(transform, rij, kol)
            e_ts = tijdstips[eerste_idx[rij, kol]].strftime("%Y-%m-%d %H:%M:%S")
            l_ts = tijdstips[laatste_idx[rij, kol]].strftime("%Y-%m-%d %H:%M:%S")
            w.writerow([rij, kol, round(x, 2), round(y, 2),
                        int(count_matrix[rij, kol]),
                        round(float(max_leq_matrix[rij, kol]), 2),
                        e_ts, l_ts])
    print(f"Locaties-CSV opgeslagen: {locaties_csv}")

    # ─── OUTPUT 4: Samenvatting ───────────────────────────────────────────────
    totaal_pixels       = RIJEN * KOLOMMEN
    pixels_overtreding  = int(np.sum(count_matrix > 0))
    max_overtredingen   = int(np.max(count_matrix))
    totaal_vensters     = N - VENSTER_SECONDEN + 1
    vensters_overtreding = len(venster_log)

    # Dagopsplitsing
    dag_stats = {}
    for ts, n in venster_log:
        dag = ts.strftime("%Y-%m-%d")
        dag_stats.setdefault(dag, {"vensters": 0, "max_pixels": 0})
        dag_stats[dag]["vensters"] += 1
        dag_stats[dag]["max_pixels"] = max(dag_stats[dag]["max_pixels"], n)

    samenvatting = output_dir / "samenvatting.txt"
    lijnen = [
        "=" * 60,
        "FESTIVAL GELUID — WETTELIJKE ANALYSE (85 dBA / 15 min)",
        "=" * 60,
        f"Periode:               {tijdstips[0]}  →  {tijdstips[-1]}",
        f"Aantal rasters:        {N}",
        f"Rastergrootte:         {RIJEN} × {KOLOMMEN} pixels (5×5 m)",
        f"Totaal pixels:         {totaal_pixels:,}",
        "",
        "─── VENSTERS ─────────────────────────────────────────",
        f"Totaal vensters (15'):  {totaal_vensters:,}",
        f"Vensters in overtreding:{vensters_overtreding:,}  ({100*vensters_overtreding/totaal_vensters:.1f}%)",
        "",
        "─── LOCATIES ─────────────────────────────────────────",
        f"Pixels ooit in overtreding: {pixels_overtreding:,}  ({100*pixels_overtreding/totaal_pixels:.2f}% v/h gebied)",
        f"Max. overschrijdingen/pixel:{max_overtredingen:,} vensters",
        "",
        "─── PER DAG ──────────────────────────────────────────",
    ]
    for dag, stats in sorted(dag_stats.items()):
        lijnen.append(
            f"  {dag}:  {stats['vensters']:>6} overtredende vensters  "
            f"(max {stats['max_pixels']:,} pixels tegelijk)"
        )
    lijnen += [
        "",
        "─── ERGSTE MOMENTEN (top 10 naar aantal pixels) ──────",
    ]
    top10 = sorted(venster_log, key=lambda x: x[1], reverse=True)[:10]
    for ts, n, max_leq in top10:
        ts_begin = ts - timedelta(seconds=VENSTER_SECONDEN - 1)
        lijnen.append(f"  {ts_begin.strftime('%Y-%m-%d %H:%M:%S')} – {ts.strftime('%H:%M:%S')}  →  {n:,} pixels  (max LAeq,15min = {max_leq:.1f} dBA)")
    lijnen += ["", "=" * 60]

    tekst = "\n".join(lijnen)
    print("\n" + tekst)
    with open(samenvatting, "w", encoding="utf-8") as f:
        f.write(tekst + "\n")
    print(f"\nSamenvatting opgeslagen: {samenvatting}")


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Optioneel: pad naar Raster_2 als argument meegeven
    # Gebruik: python analyseer_geluidsoverschrijdingen.py /pad/naar/Raster_2
    if len(sys.argv) == 2:
        DATA_DIR = Path(sys.argv[1])

    print(f"Data map:   {DATA_DIR.resolve()}")
    print(f"Output map: {OUTPUT_DIR.resolve()}")
    print()
    analyseer(DATA_DIR, OUTPUT_DIR)