import os
import glob
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from tqdm import tqdm

# ========== CONFIG ==========
INPUT_FOLDER = r"C:\Users\Gebruiker\Documents\DATA\Metingen\Interpolatietabellen\30_augustus"
OUTPUT_FOLDER = r"C:\Users\Gebruiker\Documents\DATA\Raster_2\30_augustus_dynamisch"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# CSV met meetpunten en coördinaten
POINTS_CSV = r"C:\Users\Gebruiker\Documents\DATA\Meetpunten\meetpunten_30082025.csv"

# PAD naar dynamische meter CSV (per-seconde X,Y,dBA)
DYNAMIC_CSV = r"C:\Users\Gebruiker\Documents\DATA\Metingen\Interpolatietabellen\Dynamisch\30_augustus\dynamisch_meetpunt_30augustus.csv"

# Inlezen en dictionary maken
df_points = pd.read_csv(POINTS_CSV, dtype={"id": str})
POINTS = {row["id"].strip(): (row["X"], row["Y"]) for _, row in df_points.iterrows()}

# Lees dynamische meter CSV en indexeer op datetime
df_dyn = pd.read_csv(DYNAMIC_CSV)
df_dyn["datetime"] = pd.to_datetime(df_dyn["Datum"].astype(str) + " " + df_dyn["Tijd"].astype(str), dayfirst=True, errors="coerce")
# vervang komma-decimaal en zet float
df_dyn["dBA"] = df_dyn["dBA"].astype(str).str.replace(",", ".", regex=False).astype(float, errors="ignore")
df_dyn = df_dyn.set_index("datetime")[["dBA", "X", "Y"]]

# CRS instellen op Belgian Lambert 72
CRS = "EPSG:31370"

# IDW parameters
POWER = 2.0          # inverse distance power (2 = gewone IDW)
SEARCH_RADIUS = None # in same units as coords; None = use all points
CELL_SIZE = 5.0      # raster cellsize (grid spacing) in same units as coordinates
MIN_POINTS = 6       # minimaal aantal meetpunten vereist om te interpoleren

# Zoek alle CSV-bestanden in de inputmap
all_files = glob.glob(os.path.join(INPUT_FOLDER, "*.csv"))

if not all_files:
    raise SystemExit(f"Geen CSV-bestanden gevonden in {INPUT_FOLDER}")

# =====================================

# Helper: lees één CSV en maak timeseries (verwacht kolommen 'Datum' (dd-mm-YYYY) en 'Tijd' (HH:MM:SS) of één 'Datetime' kolom)
def read_sensor_csv(path):
    df = pd.read_csv(path)

    # zoek tijdstempel
    dt = None
    if {"Datum","Tijd"}.issubset(df.columns):
        dt = pd.to_datetime(df["Datum"].astype(str) + " " + df["Tijd"].astype(str),
                            dayfirst=True, errors="coerce")
    else:
        # probeer verschillende mogelijke kolomnamen
        for col in ["Datetime", "DateTime", "timestamp", "Timestamp", "tijdstip", "tijd"]:
            if col in df.columns:
                dt = pd.to_datetime(df[col], dayfirst=True, errors="coerce")
                break
    if dt is None:
        raise ValueError(f"Geen herkenbare datum/tijd kolom in {path}")

    # zoek dBA kolom
    dba_col = [c for c in df.columns if "dba" in c.lower()]
    if dba_col:
        dba_col = dba_col[0]
    else:
        # neem laatste kolom als fallback
        dba_col = df.columns[-1]

    # meetpunt-id
    if "Meetpunt" not in df.columns:
        raise ValueError(f"Geen kolom 'Meetpunt' gevonden in {path}")
    sensor_id = str(df["Meetpunt"].iloc[0]).strip()

    # maak series
    series = pd.Series(df[dba_col].values, index=dt)
    series = series.astype(str).str.replace(",", ".").astype(float, errors="ignore")
    series.index.name = "datetime"

    return sensor_id, series


series_dict = {}
for path in all_files:
    sensor_id, s = read_sensor_csv(path)
    if sensor_id not in POINTS:
        print(f"Waarschuwing: meetpunt {sensor_id} heeft geen coördinaten in {POINTS_CSV}")
        continue
    s = s[~s.index.isna()].sort_index()
    series_dict[sensor_id] = s

print("POINTS keys:", list(POINTS.keys()))
print("Sensor IDs gevonden:", list(series_dict.keys()))


# 1) lees alle beschikbare sensorbestanden in en koppel op basis van 'Meetpunt'

series_dict = {}
sensor_to_file = {}
missing_coordinates = set()

for path in all_files:
    try:
        sensor_id, s = read_sensor_csv(path)
    except Exception as e:
        print(f"Fout bij lezen van {path}: {e}")
        continue

    # verwijder corrupte of dubbele tijdstempels
    s = s[~s.index.isna()]
    s = s.sort_index()
    s = s[~s.index.duplicated(keep="first")]

    if sensor_id not in POINTS:
        print(f"Waarschuwing: meetpunt {sensor_id} heeft geen coördinaten in {POINTS_CSV}")
        continue

    series_dict[sensor_id] = s
    sensor_to_file[sensor_id] = path  # informatief

if not series_dict:
    raise SystemExit("Geen bruikbare sensordata gevonden. Controleer kolom 'Meetpunt' en POINTS-CSV.")

available_keys = list(series_dict.keys())
print("Beschikbare sensoren die matchen met punt-coördinaten:", available_keys)

# gezamenlijke tijdrange
min_t = min(s.index.min() for s in series_dict.values())
max_t = max(s.index.max() for s in series_dict.values())
# create per-second index
time_index = pd.date_range(start=min_t.floor('S'), end=max_t.ceil('S'), freq='S')

# Build wide DataFrame: columns = sensor keys
df_all = pd.DataFrame(index=time_index)
for k, s in series_dict.items():
    # reindex to full seconds; leave NaN where missing
    df_all[k] = s.reindex(time_index).values
    # Voor meter 0: vul tussenliggende seconden op met lineaire interpolatie
    if k == "0":
        df_all[k] = df_all[k].interpolate(method='time', limit_area='inside')

# Precompute grid bounds from POINTS
xs = np.array([POINTS[k][0] for k in available_keys])
ys = np.array([POINTS[k][1] for k in available_keys])
minx, maxx = xs.min(), xs.max()
miny, maxy = ys.min(), ys.max()
# add small margin
margin = max(CELL_SIZE * 5, 0) + 150
minx -= margin; miny -= margin; maxx += margin; maxy += margin

# grid size
nx = int(np.ceil((maxx - minx) / CELL_SIZE))
ny = int(np.ceil((maxy - miny) / CELL_SIZE))
# ensure at least 1 cell
nx = max(1, nx); ny = max(1, ny)

grid_x = np.linspace(minx + CELL_SIZE/2, minx + (nx-0.5)*CELL_SIZE, nx)
grid_y = np.linspace(miny + CELL_SIZE/2, miny + (ny-0.5)*CELL_SIZE, ny)
gx, gy = np.meshgrid(grid_x, grid_y[::-1])  # row 0 = top

# prepare for writing rasters
transform = from_origin(minx, maxy, CELL_SIZE, CELL_SIZE)  # transform for rasterio

def idw_interpolate(xy, values, xi, yi, power=2.0, search_radius=None, eps=1e-8):
    """
    xy: (n,2) array of points
    values: (n,) array of values
    xi, yi: grid arrays (m,n)
    returns: grid of interpolated values
    """
    # flatten grid
    xig = xi.ravel()
    yig = yi.ravel()
    interp = np.full(xig.shape, np.nan, dtype=float)

    # compute distances from grid points to sample points
    # for memory reasons compute in chunks if grid large
    n_samples = xy.shape[0]
    n_grid = xig.size
    # if few samples: vectorize straightforwardly
    # compute distance matrix: (n_grid, n_samples)
    dx = xig[:, None] - xy[:, 0][None, :]
    dy = yig[:, None] - xy[:, 1][None, :]
    dist = np.hypot(dx, dy) + eps

    if search_radius is not None:
        mask_within = dist <= search_radius
    else:
        mask_within = np.ones_like(dist, dtype=bool)

    # weights
    with np.errstate(divide='ignore'):
        w = (1.0 / (dist ** power)) * mask_within

    # handle exact-zero distances: if any sample exactly at gridpoint, take that sample value
    zero_mask = dist <= eps*10
    any_zero = zero_mask.any(axis=1)
    interp[any_zero] = values[zero_mask[any_zero].argmax(axis=1)]

    # for others, compute weighted avg where sum(weights)>0
    sumw = w.sum(axis=1)
    nonzero = (sumw > 0) & (~any_zero)
    interp[nonzero] = (w[nonzero] * values[None, :]).sum(axis=1)[nonzero] / sumw[nonzero]

    return interp.reshape(xi.shape)

# add explicit testmode flag
TESTMODE = False

# ========== TEST INTERVAL ==========
# Stel een testperiode in (hier: 30 augustus 2025 tussen 20:00 en 20:05)
start_test = pd.Timestamp("2025-08-30 20:45:00")
end_test   = pd.Timestamp("2025-08-30 20:50:00")

# Filter de tijdindex op dit interval (wordt gebruikt als TESTMODE actief is)
times_test = df_all.index[(df_all.index >= start_test) & (df_all.index <= end_test)]

print(f"Testmodus actief: {len(times_test)} seconden geselecteerd tussen {start_test} en {end_test}")
# ===================================

# Loop over every second and generate TIFF
if TESTMODE:
    times = times_test
else:
    times = df_all.index
# optional: use tqdm for progress
for ts in tqdm(times, desc="Interpolating seconds"):
    vals = df_all.loc[ts].values  # order matches available_keys
    # prepare points with non-nan values (statische meetpunten)
    mask = ~np.isnan(vals)
    static_count = int(mask.sum())
        # controleer of de dynamische meter een meting heeft op dit tijdstip
    dyn_present = False
    if ts in df_dyn.index:
        dyn_row = df_dyn.loc[ts]
        # bij meerdere rijen op dezelfde timestamp: neem eerste
        if isinstance(dyn_row, pd.DataFrame):
            dyn_row = dyn_row.iloc[0]
        if (not pd.isna(dyn_row.get("dBA"))) and (not pd.isna(dyn_row.get("X"))) and (not pd.isna(dyn_row.get("Y"))):
            dyn_present = True
    # totaal aantal beschikbare punten (statisch + dynamisch indien aanwezig)
    total_points = static_count + (1 if dyn_present else 0)
    if total_points < MIN_POINTS:
        # skip writing raster if not enough sensors
        continue
    # verzamel statische samples
    sample_keys = list(np.array(available_keys)[mask])
    sample_vals = list(vals[mask].astype(float))
    sample_xy = [POINTS[k] for k in sample_keys]

    # voeg dynamische meter toe (indien aanwezig)
    if dyn_present:
        dyn_id = "D"
        sample_keys.append(dyn_id)
        sample_vals.append(float(dyn_row["dBA"]))
        sample_xy.append((float(dyn_row["X"]), float(dyn_row["Y"])))

    # zet om naar numpy arrays voor interpolatiefunctie
    sample_keys = np.array(sample_keys)
    sample_vals = np.array(sample_vals, dtype=float)
    sample_xy = np.array(sample_xy, dtype=float)

    # optionally prune samples outside SEARCH_RADIUS per grid center (skipped)
    grid = idw_interpolate(sample_xy, sample_vals, gx, gy, power=POWER, search_radius=SEARCH_RADIUS)

    # write GeoTIFF
    ts_str = ts.strftime("%Y%m%d_%H%M%S")
    out_name = f"{ts_str}.tif"
    out_path = os.path.join(OUTPUT_FOLDER, out_name)

    # rasterio expects (bands, rows, cols)
    profile = {
        'driver': 'GTiff',
        'height': grid.shape[0],
        'width': grid.shape[1],
        'count': 1,
        'dtype': 'float32',
        'crs': CRS,
        'transform': transform,
        'compress': 'lzw'
    }

    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(grid.astype('float32'), 1)
        # add metadata: which meters influenced this raster and how many
        meters_list = ",".join(sample_keys.tolist())
        dst.update_tags(meters_actief=meters_list, aantal_meters=str(len(sample_keys)))

print("Klaar - alle beschikbare seconden geinterpoleerd en opgeslagen.")
