from numpy import *
import matplotlib.pyplot as plt
import sys
from matplotlib.colors import LogNorm
import csv
from scipy.stats import gaussian_kde
import seaborn as sns
import numpy as np
import os
import pandas as pd
import pathlib
import random
from typing import Callable, Dict, Tuple
from tqdm import tqdm  # optional, nice progress bars


def apply_mask(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Convenient wrapper for boolean indexing."""
    return arr[mask]


def build_dataframe(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    m: np.ndarray,
    ssfr: np.ndarray,
) -> pd.DataFrame:
    """Create a pandas DataFrame with the required column order."""
    return pd.DataFrame(
        {"x": x, "y": y, "z": z, "m": m, "ssfr": ssfr}
    )


def save_parent(df: pd.DataFrame, out_path: pathlib.Path) -> None:
    """Write the full masked catalogue to CSV."""
    df.to_csv(out_path, index=False)
    print(f"✔ Saved parent catalogue → {out_path}")


def filter_category(
    df: pd.DataFrame,
    mass_key: str,
    ssfr_key: str,
) -> pd.DataFrame:
    """Return rows that satisfy both the mass and sSFR masks."""
    mass_mask = categories["m_sel"][mass_key](df["m"].values)
    ssfr_mask = categories["ssfr_sel"][ssfr_key](df["ssfr"].values)
    return df[mass_mask & ssfr_mask].reset_index(drop=True)

data=genfromtxt('merged_galaxies.txt')
x = data[:,1]
y = data[:,2]
z = data[:,3]
m = data[:,4]
ssfr = data[:,5]
stellarphotometrics_r = data[:,-3]
stellarphotometrics_i = data[:,-2]
stellarphotometrics_z = data[:,-1]
stellarphotometrics_u = data[:,-8]
stellar_photometrics_g = data[:,-4]

M_r = stellarphotometrics_r
M_i = stellarphotometrics_i
M_u = stellarphotometrics_u
M_g = stellar_photometrics_g
M_z = stellarphotometrics_z
# Assume z ~ 0.05 for the mock sample
DM = 36.7  # distance modulus at z~0.05 (Planck cosmology)

# Apparent magnitudes
m_r = M_r + DM
m_i = M_i + DM
m_u = M_u + DM
m_g = M_g + DM
m_z = M_z + DM

# Cuts

msk_r = (m_r < 22.2)
msk_i = (m_i < 21.4)
msk_u = (m_u < 22.0)
msk_g = (m_g < 22.3)
msk_z = (m_z < 20.5)
msk_ssfr = (ssfr > 0)

msk_tot = msk_r & msk_i & msk_u & msk_g & msk_z & msk_ssfr

m_sel = log10(m[msk_tot])
ssfr_sel = log10(ssfr[msk_tot])
x_sel = x[msk_tot]
y_sel = y[msk_tot]
z_sel = z[msk_tot]


parent_df = build_dataframe(x_sel, y_sel, z_sel, m_sel, ssfr_sel)

# -----------------------------------------------------------------
# Write the parent CSV
# -----------------------------------------------------------------
OUT_DIR = pathlib.Path(".")  # you can change this if you want a sub‑folder
parent_path = OUT_DIR / "random_parent.csv"
save_parent(parent_df, parent_path)

CategoryFunc = Callable[[np.ndarray], np.ndarray]

categories: Dict[str, Dict[str, CategoryFunc]] = {
    "m_sel": {
        "dwarf": lambda x: x < 9.5,
        "average": lambda x: (x >= 9.5) & (x < 10.5),
        "massive": lambda x: x >= 10.5,
    },
    "ssfr_sel": {
        "passive": lambda x: x < -10.5,
        "average": lambda x: (x >= -10.5) & (x < -9.5),
        "starburst": lambda x: x >= -9.5,
    },
}

# -----------------------------------------------------------------
# 4️⃣  Create the 9 category CSV files
# -----------------------------------------------------------------
category_paths: Dict[Tuple[str, str], pathlib.Path] = {}

for mass_cat in categories["m_sel"]:
    for ssfr_cat in categories["ssfr_sel"]:
        cat_df = filter_category(parent_df, mass_cat, ssfr_cat)

        # Skip empty categories – they would otherwise create empty files
        if cat_df.empty:
            print(f"⚠️  Category {mass_cat}/{ssfr_cat} is empty → skipped")
            continue

        file_name = f"random_{mass_cat}_{ssfr_cat}.csv"
        out_path = OUT_DIR / file_name
        cat_df.to_csv(out_path, index=False)
        category_paths[(mass_cat, ssfr_cat)] = out_path
        print(f"✔ Saved category {mass_cat}/{ssfr_cat} → {out_path}")


# -----------------------------------------------------------------
# 5️⃣  Bootstrap sub‑samples (50 % of rows, 100 repeats)
# -----------------------------------------------------------------
BOOTSTRAP_DIR = OUT_DIR / "bootstraps"
BOOTSTRAP_DIR.mkdir(exist_ok=True)

# reproducibility – change the seed if you want a different set
RNG_SEED = 12345
rng = np.random.default_rng(RNG_SEED)

N_BOOT = 100          # number of bootstrap realisations per category
FRAC = 0.5            # fraction of rows to keep each time

for (mass_cat, ssfr_cat), cat_path in tqdm(
    category_paths.items(),
    desc="Bootstrapping categories",
    unit="category",
):
    cat_df = pd.read_csv(cat_path)  # read back – cheap for the sizes we have
    n_rows = len(cat_df)

    if n_rows == 0:
        # Should never happen because we filtered empties above
        continue

    # Number of rows to draw each time (rounded to nearest integer)
    n_draw = int(round(FRAC * n_rows))
    if n_draw == 0:
        print(
            f"⚠️  Category {mass_cat}/{ssfr_cat} has too few rows "
            f"({n_rows}) to draw 50 % → skipped"
        )
        continue

    for i in range(N_BOOT):
        # Random 50 % *without* replacement (the usual “sub‑sample” bootstrap)
        # If you want the classic bootstrap (with replacement) replace the line
        # below with: idx = rng.integers(0, n_rows, size=n_draw)
        idx = rng.choice(n_rows, size=n_draw, replace=False)

        boot_df = cat_df.iloc[idx].reset_index(drop=True)

        boot_name = f"random_{mass_cat}_{ssfr_cat}_{i+1}.csv"
        boot_path = BOOTSTRAP_DIR / boot_name
        boot_df.to_csv(boot_path, index=False, header=False)