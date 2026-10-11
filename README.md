# AI-Based Weather Rainfall Prediction System
### Physics-Guided Machine Learning for Extreme Precipitation Estimation — Cyclone Fani (2019)

---

## 📌 Overview

This repository contains the complete research project for:

> **"Physics-Guided Machine Learning for Extreme Precipitation Estimation During Tropical Cyclone Fani (2019)"**
> *Submitted for IEEE Transactions on Geoscience and Remote Sensing (TGRS)*

The study presents a **Physics-Guided Machine Learning (PGML) Ensemble** framework designed to overcome the tail-risk underestimation problem in cyclone precipitation modelling — applied exclusively to **Extremely Severe Cyclonic Storm Fani (April 25 – May 05, 2019)** over the Bay of Bengal.

---

## 🔑 Key Results

| Metric | Baseline (MSE) | PGML Ensemble (FISW) |
|:---|:---:|:---:|
| Domain MAE (mm/h) | 0.202 | 0.289 |
| Eyewall Recall ≥15 mm/h | 23.33% | **87.93%** |
| Critical Success Index | 21.10% | **62.96%** |
| F1-Score @ ≥15 mm/h | — | **77.27%** |
| Domain Accuracy | — | **96.58%** |

> The PGML model deliberately sacrifices average-field MAE to achieve a **4× improvement** in extreme eyewall detection — the key scientific contribution.

---

## 📂 Repository Structure

```
Fani_Rainfall_Prediction/
│
├── src/                            # Core source modules
│   ├── config.py                   # Configuration & paths
│   ├── data_loader.py              # ERA5 NetCDF loading & deaccumulation
│   ├── features.py                 # 55-variable physics-guided feature engineering
│   ├── fisw_loss.py                # FISW Loss implementation (the key contribution)
│   ├── models.py                   # ML model suite (RF, ET, XGB, LGBM, CatBoost)
│   ├── evaluation.py               # Metrics: MAE/RMSE/extreme recall/CSI/CI
│   ├── pipeline.py                 # End-to-end training + evaluation pipeline
│   ├── diagnostics.py              # All figure generation functions
│   └── leakage_audit.py            # Temporal leakage verification
│
├── canonical_experiment.py         # Master canonical experiment runner
├── run_pipeline.py                 # Simple pipeline entry point
├── fani_official_besttrack_IMD.csv # IMD Best Track data for Fani
├── FINAL_IEEE_MANUSCRIPT.pdf       # Camera-ready IEEE manuscript
└── README.md
```

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install numpy pandas xarray xgboost lightgbm catboost scikit-learn reportlab scipy
```

### 2. Download required data
- **ERA5:** Download via [CDS API](https://cds.climate.copernicus.eu/) for domain 78°E–95°E, 1°N–28°N, April 25 – May 05, 2019
- **GPM IMERG V07B Final:** Available from [NASA GES DISC](https://disc.gsfc.nasa.gov/)
- Place NetCDF files in `Fani_Rainfall_Prediction/`

### 3. Run canonical experiment
```bash
cd Fani_Rainfall_Prediction
python canonical_experiment.py
```

This generates all CSV metrics, confidence intervals, and the final IEEE PDF automatically.

---

## 🧠 Key Methodological Contributions

### 1. Physics-Guided Feature Engineering (55 variables)
- Storm-relative geometry (Haversine distance, azimuth, along/cross-track)
- Moisture flux divergence: `−∇·(qV)`
- Spatial Laplacian & gradient operators on rainfall fields
- Multi-scale temporal lag features (t−1h to t−12h)
- Instability indices (CAPE, TCWV, vertical wind shear)

### 2. Focal Intensity Sample Weighting (FISW) Loss
```
w(y_i) = 1 + α·(y_i / y_max)^γ      [α=15, γ=2]
L_FISW = (1/N) Σ [ w(y_i)|y_i - ŷ_i| + λ·max(0, y_i - ŷ_i)² ]   [λ=0.5]
```
Applied as `sample_weight` in tree ensemble training.

### 3. Validation-Error Weighted Ensemble
Ensemble weights derived exclusively from **validation predictions** — test set never used for weight selection.

---

## ⚠️ Scope & Limitations

- **Single-storm study:** Evaluated exclusively on Cyclone Fani (2019). Generalizability to other cyclones is not established.
- **Retrospective estimation:** Uses contemporaneous ERA5 fields. Not an operational real-time forecast.
- **IMERG reference:** IMERG Final incorporates ancillary ERA5 inputs — treated as an *external reference product*, not independent ground truth.

---

## 📄 Publication

**Paper:** `FINAL_IEEE_MANUSCRIPT.pdf` (included in root directory)  
**Target Journal:** IEEE Transactions on Geoscience and Remote Sensing (TGRS)

---

## 👤 Author

**Sudhanshu Jha**  
GitHub: [@sudhanshu-jha45](https://github.com/sudhanshu-jha45)
