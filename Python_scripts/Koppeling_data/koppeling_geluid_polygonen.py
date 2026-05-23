import geopandas as gpd
import rasterio
from rasterio.mask import mask
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import warnings
import time 
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATIE
# ============================================================================

TEST_MODE = False 
N_TEST_POLYGONS = 10

PEAK_THRESHOLD = 65  # dB(A)
MIN_GAP_SECONDS = 5  # Minimale gap tussen events

# PADEN
SHAPEFILE_PATH = r"C:\Users\Gebruiker\Documents\DATA\Tabel_voor_QGIS\912_vooranalyse.shp" 
BLOK_RASTERS_DIR = r"C:\Users\Gebruiker\Documents\DATA\Aggregatie_blokken"
SEC_RASTERS_DIR = r"C:\Users\Gebruiker\Documents\DATA\Raster_2"
OUTPUT_CSV = r"C:\Users\Gebruiker\Documents\DATA\Programma\resultaten_analyse.csv"

# REFERENTIE RASTER VOOR OVERLAP CHECK
REFERENCE_RASTER = r"C:\Users\Gebruiker\Documents\DATA\Raster_2\30_augustus\1sec\20250830_193745.tif"

# KOLOM HERNOEMEN (pas aan naar wens, of laat zoals het is)
RENAME_COLUMNS = {
    'visitor_id': 'id',
    'Tabel_gesl': 'geslacht',
    'Tabel_leef': 'leeftijd',
    'Tabel_stan': 'stance',
    'Tabel_st_1': 'stance_numeriek',
    'oppervlakt': 'oppervlakte',
    'afstand': 'afstand',
    'richting': 'richting'
}

# BLOK DEFINITIES
BLOCKS = {
    '29aug': [
        {'blok': 1, 'date': '20250829', 'start': '15:00:00', 'end': '17:15:00'},
        {'blok': 2, 'date': '20250829', 'start': '17:15:00', 'end': '20:30:00'}
    ],
    '30aug': [
        {'blok': 1, 'date': '20250830', 'start': '15:00:00', 'end': '17:15:00'},
        {'blok': 2, 'date': '20250830', 'start': '17:15:00', 'end': '20:30:00'},
        {'blok': 3, 'date': '20250830', 'start': '20:30:00', 'end': '22:15:00'},
        {'blok': 4, 'date': '20250830', 'start': '22:15:00', 'end': '23:59:59'}
    ],
    '31aug': [
        {'blok': 1, 'date': '20250831', 'start': '15:00:00', 'end': '17:15:00'},
        {'blok': 2, 'date': '20250831', 'start': '17:15:00', 'end': '20:30:00'},
        {'blok': 3, 'date': '20250831', 'start': '20:30:00', 'end': '22:15:00'},
        {'blok': 4, 'date': '20250831', 'start': '22:15:00', 'end': '23:59:59'}
    ]
}

# ============================================================================
# HULPFUNCTIES
# ============================================================================

def check_polygon_overlap(polygon_geom, reference_raster_path):
    """
    Controleer of polygoon overlapt met het referentie raster.
    Retourneert True als er overlap is, False indien niet.
    """
    try:
        with rasterio.open(reference_raster_path) as src:
            # Probeer data te extraheren
            out_image, out_transform = mask(src, [polygon_geom], crop=True, nodata=np.nan)
            data = out_image[0]
            
            # Check of er valide data is
            valid_data = data[~np.isnan(data)]
            
            if len(valid_data) > 0:
                return True
            else:
                return False
    except Exception as e:
        # Bij fout (bijv. geen overlap) → False
        return False

def time_to_seconds(time_str):
    """Converteer HH:MM:SS naar seconden sinds middernacht"""
    h, m, s = map(int, time_str.split(':'))
    return h * 3600 + m * 60 + s

def get_second_rasters_for_block(day_key, block_info):
    """Vind alle seconde-rasters voor een specifiek blok"""
    date = block_info['date']
    start_sec = time_to_seconds(block_info['start'])
    end_sec = time_to_seconds(block_info['end'])
    
    # Pad naar seconde-rasters
    sec_dir = Path(SEC_RASTERS_DIR) / f"{day_key.split('aug')[0]}_augustus" / "1sec"
    
    if not sec_dir.exists():
        print(f"  ⚠️  Directory niet gevonden: {sec_dir}")
        return []
    
    # Vind alle rasters binnen tijdvenster
    rasters = []
    for raster_file in sorted(sec_dir.glob(f"{date}_*.tif")):
        # Extract tijd uit filename: YYYYMMDD_HHMMSS.tif
        time_part = raster_file.stem.split('_')[1]  # HHMMSS
        h = int(time_part[0:2])
        m = int(time_part[2:4])
        s = int(time_part[4:6])
        file_sec = h * 3600 + m * 60 + s
        
        if start_sec <= file_sec <= end_sec:
            rasters.append(raster_file)
    
    return rasters

def get_block_raster_path(day_key, block_info):
    """Vind het geaggregeerde blok-raster"""
    date = block_info['date']
    blok_num = block_info['blok']
    
    # Zoek in juiste dag directory
    day_dir = Path(BLOK_RASTERS_DIR) / f"{day_key.split('aug')[0]}_augustus"
    
    # Verwachte filename: YYYYMMDD_blokX.tif
    raster_name = f"{date}_blok{blok_num}.tif"
    raster_path = day_dir / raster_name
    
    if raster_path.exists():
        return raster_path
    else:
        print(f"  ⚠️  Blok-raster niet gevonden: {raster_path}")
        return None

def extract_polygon_data(polygon_geom, raster_path):
    """Extraheer alle pixelwaarden binnen een polygoon uit een raster"""
    try:
        with rasterio.open(raster_path) as src:
            # Mask raster met polygoon
            out_image, out_transform = mask(src, [polygon_geom], crop=True, nodata=np.nan)
            data = out_image[0]  # Eerste band
            
            # Verwijder nodata waarden
            valid_data = data[~np.isnan(data)]
            return valid_data
    except Exception as e:
        print(f"    ⚠️  Fout bij extractie uit {raster_path.name}: {e}")
        return np.array([])

def detect_events(timeseries, threshold, min_gap):
    """
    Detecteer piek-events in een tijdreeks.
    
    Event = groep van opeenvolgende waarden >threshold,
    gescheiden door minimaal min_gap waarden <=threshold
    """
    if len(timeseries) == 0:
        return 0
    
    # Binaire reeks: 1 = boven drempel, 0 = onder drempel
    binary = (timeseries > threshold).astype(int)
    
    # Detecteer overgangen
    events = 0
    in_event = False
    gap_counter = 0
    
    for val in binary:
        if val == 1:  # Boven drempel
            if not in_event:
                events += 1
                in_event = True
            gap_counter = 0
        else:  # Onder drempel
            gap_counter += 1
            if gap_counter >= min_gap:
                in_event = False
    
    return events

def calculate_metrics_for_polygon(polygon_geom, day_key, block_info):
    """Bereken alle 4 metrieken voor één polygoon voor één blok"""
    
    metrics = {
        'mean': np.nan,
        'max': np.nan,
        'peak_intensity': np.nan,
        'peak_events': np.nan
    }
    
    # 1. GEMIDDELDE - van geaggregeerd blok-raster
    blok_raster = get_block_raster_path(day_key, block_info)
    if blok_raster and blok_raster.exists():
        data = extract_polygon_data(polygon_geom, blok_raster)
        if len(data) > 0:
            metrics['mean'] = np.mean(data)
    
    # 2-4. Metrieken van seconde-rasters
    sec_rasters = get_second_rasters_for_block(day_key, block_info)
    
    if len(sec_rasters) == 0:
        return metrics
    
    # Verzamel data per pixel over alle seconden
    # pixel_timeseries[i] = array van alle waarden voor pixel i over tijd
    pixel_timeseries = []
    
    # Lees eerste raster om pixellocaties te bepalen
    first_data = extract_polygon_data(polygon_geom, sec_rasters[0])
    n_pixels = len(first_data)
    
    if n_pixels == 0:
        return metrics
    
    # Initialiseer timeseries per pixel
    pixel_timeseries = [[] for _ in range(n_pixels)]
    
    # Verzamel waarden over tijd
    for raster_path in sec_rasters:
        data = extract_polygon_data(polygon_geom, raster_path)
        if len(data) == n_pixels:
            for i, val in enumerate(data):
                pixel_timeseries[i].append(val)
    
    # Converteer naar numpy arrays
    pixel_timeseries = [np.array(ts) for ts in pixel_timeseries if len(ts) > 0]
    
    if len(pixel_timeseries) == 0:
        return metrics
    
    # 2. MAXIMUM
    pixel_maxima = [np.max(ts) for ts in pixel_timeseries]
    metrics['max'] = np.max(pixel_maxima)
    
    # 3. PIEKINTENSITEIT (mediaan van pixel-percentages)
    n_seconds = len(pixel_timeseries[0])
    pixel_peak_pcts = [(np.sum(ts > PEAK_THRESHOLD) / n_seconds * 100) 
                       for ts in pixel_timeseries]
    metrics['peak_intensity'] = np.median(pixel_peak_pcts)
    
    # 4. AANTAL PIEKEN (mediaan van pixel-event counts)
    pixel_event_counts = [detect_events(ts, PEAK_THRESHOLD, MIN_GAP_SECONDS) 
                          for ts in pixel_timeseries]
    metrics['peak_events'] = np.median(pixel_event_counts)
    
    return metrics

# ============================================================================
# HOOFDPROGRAMMA
# ============================================================================

def main():
    start_time = time.time()  # TOEGEVOEGD: Start timer
    
    print("=" * 80)
    print("RUIMTELIJKE GELUIDSANALYSE - STADSFESTIVAL")
    print("=" * 80)
    print(f"\nTestmodus: {'JA' if TEST_MODE else 'NEE'}")
    if TEST_MODE:
        print(f"Aantal te analyseren polygonen: {N_TEST_POLYGONS}")
    print(f"Piekdrempel: {PEAK_THRESHOLD} dB(A)")
    print(f"Minimale gap tussen events: {MIN_GAP_SECONDS} seconden\n")
    
    # Laad shapefile
    print(f"📂 Shapefile inladen: {SHAPEFILE_PATH}")
    gdf = gpd.read_file(SHAPEFILE_PATH)
    print(f"   ✓ {len(gdf)} polygonen gevonden")
    print(f"   ✓ CRS: {gdf.crs}")
    
    # Controleer verwachte kolommen
    expected_cols = list(RENAME_COLUMNS.keys())
    missing = [col for col in expected_cols if col not in gdf.columns]
    if missing:
        print(f"   ⚠️  Ontbrekende kolommen: {missing}")
    
    # Testmodus: selecteer subset
    if TEST_MODE:
        gdf = gdf.head(N_TEST_POLYGONS).copy()
        print(f"   ℹ️  Testmodus: eerste {N_TEST_POLYGONS} polygonen geselecteerd")
    
    # Hernoem kolommen
    gdf = gdf.rename(columns=RENAME_COLUMNS)
    
    # Bereid output dataframe voor
    result_df = gdf[list(RENAME_COLUMNS.values())].copy()
    
    # Genereer kolomnamen voor alle metrieken
    all_columns = []
    for day_key, blocks in BLOCKS.items():
        for block_info in blocks:
            blok_num = block_info['blok']
            all_columns.extend([
                f"G_{day_key}_blok{blok_num}",   # Gemiddelde
                f"M_{day_key}_blok{blok_num}",   # Maximum
                f"PI_{day_key}_blok{blok_num}",  # Piekintensiteit
                f"PM_{day_key}_blok{blok_num}"   # Piekmomenten
            ])
    
    # Initialiseer kolommen met NaN
    for col in all_columns:
        result_df[col] = np.nan
    
    print(f"\n🔄 Start analyse voor {len(gdf)} polygonen over {sum(len(b) for b in BLOCKS.values())} blokken...\n")
    
    # NIEUWE CODE: Check referentie raster
    print(f"🗺️  Controleer overlap met referentie raster...")
    print(f"   Referentie: {Path(REFERENCE_RASTER).name}\n")
    
    # Hoofdloop: per polygoon, per dag, per blok
    polygons_with_overlap = 0
    polygons_without_overlap = 0
    
    for idx, row in gdf.iterrows():
        polygon_geom = row.geometry.__geo_interface__
        
        print(f"Polygoon {idx + 1}/{len(gdf)}", end=" ")
        
        # NIEUWE CODE: Check overlap met referentie raster
        has_overlap = check_polygon_overlap(polygon_geom, REFERENCE_RASTER)
        
        if not has_overlap:
            print("❌ Geen overlap met rasters - alle metrieken leeg gelaten")
            polygons_without_overlap += 1
            continue  # Skip alle berekeningen voor deze polygoon
        
        print("✓ Overlap aanwezig")
        polygons_with_overlap += 1
        
        for day_key, blocks in BLOCKS.items():
            for block_info in blocks:
                blok_num = block_info['blok']
                
                print(f"  📊 {day_key} blok {blok_num} ({block_info['start'][:5]}-{block_info['end'][:5]})...", end=" ")
                
                # Bereken metrieken
                metrics = calculate_metrics_for_polygon(polygon_geom, day_key, block_info)
                
                # Sla resultaten op
                result_df.at[idx, f"G_{day_key}_blok{blok_num}"] = metrics['mean']
                result_df.at[idx, f"M_{day_key}_blok{blok_num}"] = metrics['max']
                result_df.at[idx, f"PI_{day_key}_blok{blok_num}"] = metrics['peak_intensity']
                result_df.at[idx, f"PM_{day_key}_blok{blok_num}"] = metrics['peak_events']
                
                # Status feedback
                if not np.isnan(metrics['mean']):
                    print(f"✓ (G={metrics['mean']:.1f} dB, M={metrics['max']:.1f} dB)")
                else:
                    print("⚠️  Geen data")
        
        print()
    
    # Exporteer naar CSV
    print(f"\n💾 Resultaten wegschrijven naar: {OUTPUT_CSV}")
    result_df.to_csv(OUTPUT_CSV, index=False)
    print(f"   ✓ {len(result_df)} rijen × {len(result_df.columns)} kolommen")
    
    # Samenvatting
    print("\n" + "=" * 80)
    print("ANALYSE VOLTOOID")
    print("=" * 80)
    print(f"Output: {OUTPUT_CSV}")
    print(f"Polygonen verwerkt: {len(result_df)}")
    print(f"  - Met overlap: {polygons_with_overlap}")
    print(f"  - Zonder overlap: {polygons_without_overlap}")
    print(f"Metrieken per polygoon: {len(all_columns)}")
    
    # Quick stats
    mean_cols = [col for col in result_df.columns if col.startswith('G_')]
    if mean_cols:
        overall_mean = result_df[mean_cols].mean().mean()
        print(f"\nGemiddeld geluidsniveau over alle metingen: {overall_mean:.1f} dB(A)")
    
    peak_cols = [col for col in result_df.columns if col.startswith('PM_')]
    if peak_cols:
        total_events = result_df[peak_cols].sum().sum()
        print(f"Totaal aantal piekmomenten gedetecteerd: {total_events:.0f}")
    
    # TOEGEVOEGD: Tijdstatistieken
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"\n⏱️  Uitvoeringstijd:")
    print(f"   Totaal: {elapsed_time:.1f} seconden ({elapsed_time/60:.1f} minuten)")
    if polygons_with_overlap > 0:
        print(f"   Per polygoon (met overlap): {elapsed_time/polygons_with_overlap:.1f} seconden")
    
    # Schatting voor volledige dataset
    if TEST_MODE and polygons_with_overlap > 0:
        # Laad volledig shapefile om totaal te tellen
        gdf_full = gpd.read_file(SHAPEFILE_PATH)
        total_polygons = len(gdf_full)
        
        # Schat percentage met overlap
        overlap_rate = polygons_with_overlap / len(gdf)
        estimated_with_overlap = int(total_polygons * overlap_rate)
        
        estimated_total = (elapsed_time / polygons_with_overlap) * estimated_with_overlap
        print(f"\n📊 Schatting voor {total_polygons} polygonen:")
        print(f"   Geschat aantal met overlap: ~{estimated_with_overlap}")
        print(f"   Geschatte tijd: ~{estimated_total/60:.1f} minuten (~{estimated_total/3600:.1f} uur)")

if __name__ == "__main__":
    main()