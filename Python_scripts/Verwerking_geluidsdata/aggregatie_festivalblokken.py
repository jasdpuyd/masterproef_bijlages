import os
import re
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from datetime import datetime, timedelta, time
from pathlib import Path
from collections import defaultdict

BASE_DIR   = Path(r"C:\Users\Gebruiker\Documents\DATA\Raster_2")
OUTPUT_DIR = Path(r"C:\Users\Gebruiker\Documents\DATA\Aggregatie_blokken")

REEKSEN = ["29_augustus", "30_augustus", "31_augustus"]

FESTIVALBLOKKEN = [
    (time(15, 0), time(17, 15), "blok1_15u00-17u15"),
    (time(17, 15), time(20, 30), "blok2_17u15-20u30"),
    (time(20, 30), time(22, 15), "blok3_20u30-22u15"),
    (time(22, 15), time(23, 59, 59), "blok4_22u15-einde"),
]

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


def bepaal_festivalblok(dt: datetime):
    """
    Bepaal in welk festivalblok een datetime valt.
    Geeft tuple terug: (bloknaam, start_time, eind_time) of None
    """
    tijd = dt.time()
    
    for start, eind, naam in FESTIVALBLOKKEN:
        if start <= tijd < eind:
            return (naam, start, eind)
    
    return None


def groepeer_op_festivalblok(bestanden: list[tuple[datetime, Path]]):
    """
    Groepeer (datetime, pad)-paren per festivalblok per dag.
    Geeft dict terug: (datum, bloknaam) -> [pad, ...]
    """
    groepen = defaultdict(list)

    for dt, pad in bestanden:
        blok_info = bepaal_festivalblok(dt)
        if blok_info is None:
            print(f"  ⚠ Bestand buiten festivalblokken: {pad.name} ({dt.time()})")
            continue
        
        bloknaam, _, _ = blok_info
        # Groepeer per datum en blok
        datum = dt.date()
        sleutel = (datum, bloknaam)
        groepen[sleutel].append(pad)

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
    """Verwerk alle TIF-bestanden in één reeksmap voor alle festivalblokken."""

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

    # Groepeer op festivalblok
    print(f"  → Aggregatie per festivalblok …")
    groepen = groepeer_op_festivalblok(bestanden)
    
    uitvoer_map = output_reeks_map
    uitvoer_map.mkdir(parents=True, exist_ok=True)

    for (datum, bloknaam), paden in sorted(groepen.items()):
        print(f"     - {datum} {bloknaam}: {len(paden)} bestanden")
        
        # Laad alle rasters in het blok
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

        # Stel bestandsnaam samen: YYYYMMDD_<bloknaam>.tif
        datum_str = datum.strftime("%Y%m%d")
        uitvoernaam = f"{datum_str}_{bloknaam}.tif"
        uitvoerpad  = uitvoer_map / uitvoernaam

        schrijf_raster(resultaat, profiel, uitvoerpad)

    print(f"     {len(groepen)} rasters weggeschreven naar {uitvoer_map}")


def main():
    print("=" * 60)
    print("LAeq Rasteraggregatie per Festivalblok")
    print("=" * 60)
    print("\nFestivalblokken:")
    for start, eind, naam in FESTIVALBLOKKEN:
        print(f"  - {naam}: {start.strftime('%H:%M')} - {eind.strftime('%H:%M')}")
    print()

    for reeks in REEKSEN:
        reeks_map = BASE_DIR / reeks / "1sec"
        if not reeks_map.exists():
            print(f"\n⚠ Map niet gevonden, overgeslagen: {reeks_map}")
            continue

        output_reeks_map = OUTPUT_DIR / reeks
        print(f"\n[{reeks.upper()}] Verwerken van {reeks_map} …")
        verwerk_reeks(reeks_map, output_reeks_map)

    print("\n✓ Klaar!")


if __name__ == "__main__":
    main()