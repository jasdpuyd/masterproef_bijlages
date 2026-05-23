import os
import re
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

BASE_DIR   = Path(r"C:\Users\Gebruiker\Documents\DATA\Raster_2")
OUTPUT_DIR = Path(r"C:\Users\Gebruiker\Documents\DATA\Aggregatie_2")

REEKSEN = ["29_augustus", "30_augustus", "31_augustus"]

PERIODES = {
    "1min":  1,
    "5min":  5,
    "15min": 15,
}

BESTANDSPATROON = re.compile(r"^(\d{8})_(\d{6})\.tif$", re.IGNORECASE)
# ─────────────────────────────────────────────

def parse_bestandsnaam(bestandsnaam: str):
    """Haal datum en tijd op uit de bestandsnaam. Geeft datetime terug of None."""
    m = BESTANDSPATROON.match(bestandsnaam)
    if not m:
        return None
    datum_str, tijd_str = m.group(1), m.group(2)
    return datetime.strptime(datum_str + tijd_str, "%Y%m%d%H%M%S")


def laeq(arrays: list[np.ndarray]) -> np.ndarray:
    """
    Bereken LAeq over een lijst van dB(A)-rasters.
    LAeq = 10 * log10( (1/N) * sum(10^(Li/10)) )
    NoData-pixels (NaN) worden per pixel overgeslagen.
    """
    # Stapel tot 3D array: (N, rijen, kolommen)
    stapel = np.stack(arrays, axis=0).astype(np.float64)

    # Zet NaN opzij via masked array
    masked = np.ma.masked_invalid(stapel)

    # Lineaire intensiteit
    intensiteit = np.power(10.0, masked / 10.0)

    # Gemiddelde per pixel (negeert NaN automatisch)
    gem = intensiteit.mean(axis=0)

    # Terug naar dB; vul NaN in waar geen geldige waarden waren
    resultaat = 10.0 * np.ma.log10(gem)
    return resultaat.filled(np.nan)


def groepeer_op_interval(bestanden: list[tuple[datetime, Path]], minuten: int):
    """
    Groepeer (datetime, pad)-paren in niet-overlappende intervallen.
    Elk interval start op een veelvoud van `minuten` (t.o.v. middernacht).
    Geeft dict terug: intervalstart -> [pad, ...]
    """
    groepen = defaultdict(list)
    interval = timedelta(minutes=minuten)

    for dt, pad in bestanden:
        # Rond af naar beneden naar het dichtstbijzijnde interval
        seconden_in_dag = dt.hour * 3600 + dt.minute * 60 + dt.second
        interval_seconden = minuten * 60
        afgeronde_seconden = (seconden_in_dag // interval_seconden) * interval_seconden
        intervalstart = dt.replace(
            hour=afgeronde_seconden // 3600,
            minute=(afgeronde_seconden % 3600) // 60,
            second=0,
            microsecond=0,
        )
        groepen[intervalstart].append(pad)

    return groepen


def schrijf_raster(data: np.ndarray, profiel: dict, uitvoerpad: Path):
    """Schrijf een 2D numpy-array weg als GeoTIFF."""
    uitvoerpad.parent.mkdir(parents=True, exist_ok=True)
    profiel = profiel.copy()
    profiel.update(
        dtype=rasterio.float32,
        count=1,
        compress="lzw",
        nodata=np.nan,
    )
    with rasterio.open(uitvoerpad, "w", **profiel) as dst:
        dst.write(data.astype(np.float32), 1)


def verwerk_reeks(reeks_map: Path, output_reeks_map: Path):
    """Verwerk alle TIF-bestanden in één reeksmap voor alle aggregatieperiodes."""

    # Verzamel alle geldige bestanden
    bestanden = []
    for bestand in sorted(reeks_map.iterdir()):
        if bestand.suffix.lower() not in (".tif", ".tiff"):
            continue
        dt = parse_bestandsnaam(bestand.name)
        if dt is None:
            print(f"  ⚠ Overgeslagen (naam herkend niet): {bestand.name}")
            continue
        bestanden.append((dt, bestand))

    if not bestanden:
        print(f"  Geen geldige TIF-bestanden gevonden in {reeks_map}")
        return

    print(f"  {len(bestanden)} bestanden gevonden.")

    # Lees het ruimtelijk profiel van het eerste bestand
    with rasterio.open(bestanden[0][1]) as src:
        profiel = src.profile

    # Verwerk per aggregatieperiode
    for periode_naam, minuten in PERIODES.items():
        print(f"  → Aggregatie {periode_naam} …")
        groepen = groepeer_op_interval(bestanden, minuten)
        uitvoer_map = output_reeks_map / periode_naam
        uitvoer_map.mkdir(parents=True, exist_ok=True)

        for intervalstart, paden in sorted(groepen.items()):
            # Laad alle rasters in het interval
            arrays = []
            for pad in paden:
                with rasterio.open(pad) as src:
                    arr = src.read(1).astype(np.float64)
                    nodata = src.nodata
                    if nodata is not None:
                        arr[arr == nodata] = np.nan
                    arrays.append(arr)

            if not arrays:
                continue

            # Bereken LAeq
            resultaat = laeq(arrays)

            # Stel bestandsnaam samen: YYYYMMDD_HHMMSS_<periode>.tif
            datum_str = intervalstart.strftime("%Y%m%d")
            tijd_str  = intervalstart.strftime("%H%M%S")
            uitvoernaam = f"{datum_str}_{tijd_str}_{periode_naam}.tif"
            uitvoerpad  = uitvoer_map / uitvoernaam

            schrijf_raster(resultaat, profiel, uitvoerpad)

        print(f"     {len(groepen)} rasters weggeschreven naar {uitvoer_map}")


def main():
    print("=" * 60)
    print("LAeq Rasteraggregatie")
    print("=" * 60)

    for reeks in REEKSEN:
        reeks_map = BASE_DIR / reeks
        if not reeks_map.exists():
            print(f"\n⚠ Map niet gevonden, overgeslagen: {reeks_map}")
            continue

        output_reeks_map = OUTPUT_DIR / reeks
        print(f"\n[{reeks.upper()}] Verwerken van {reeks_map} …")
        verwerk_reeks(reeks_map, output_reeks_map)

    print("\n✓ Klaar!")


if __name__ == "__main__":
    main()