# RocketDAQAnalyzer

> The developed application constitutes an integral part of the engineering thesis entitled
> **"Digital filtering of data from pressure sensors in the rocket engine system."**
> The software was created to support the analysis and post-processing of experimental data acquired
> during rocket engine test campaigns and serves as a practical implementation of the methods discussed
> in this work.



RocketDAQAnalyzer is a Python desktop application for post-processing and analysis of telemetry data
acquired from rocket engine static fire test campaigns. It ingests raw data exported from MongoDB
(BSON format) or pre-converted CSV files, applies signal processing and digital filtering pipelines,
performs frequency-domain analysis, and produces publication-quality plots.

## Features

- **BSON / MongoDB ingestion** — parse raw MongoDB exports or connect live to a test-campaign database
- **Flexible data loading** — interpolated or None-filled missing-value handling on import
- **Data transformation** — normalize, scale, offset, sign-flip, sort, rename, drop columns/rows
- **Digital filtering pipeline** — queue and apply multiple filters in sequence per channel
  - Low-pass Butterworth (IIR, zero-phase)
  - Adaptive Kalman filter (constant and constant-velocity models)
  - Wavelet denoising (PyWavelets, configurable level and thresholding mode)
  - Rolling mean and rolling median
  - Threshold clipping, negative/positive removal
- **Frequency-domain analysis** — FFT (periodogram) and Spectrogram (STFT) with configurable parameters
- **Advanced plotting** — dual Y-axes, line/scatter, custom colors, reference lines, time offset
- **Dataset synchronization** — align two datasets by peak detection for ignition-relative time axis
- **Non-blocking UI** — all heavy operations run in background threads via Qt thread pool

---

## Requirements

- Python 3.10+
- `pip`

| Package | Version |
|---|---|
| PyQt6 | ~6.8.1 |
| matplotlib | ~3.7.0 |
| numpy | ~1.26.4 |
| pandas | ~2.2.3 |
| scipy | ~1.15.2 |
| dask\[dataframe\] | ~2024.10.0 |
| PyWavelets | ~1.8.0 |
| pymongo | ~4.15.1 |
| AHRS | ~0.3.1 |

---

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/AndrewJHK/postprocessing-app
   cd postprocessing-app
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**

   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - macOS / Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

5. **Run the application**

   ```bash
   python main.py
   ```

---

## Application Overview

The application is organized into four panels, designed to be used in sequence.

### Panel 1 — Load Data

Import data into the application session.

- **Convert BSON to CSV** — select a `.bson` file exported from MongoDB; choose interpolation mode:
  - *Interpolated* — each channel carries forward its last known value for every timestamp
  - *None filled* — missing values are left as `None`
- **Load CSV** — load an already-converted CSV file directly
- **Connect to MongoDB** — connect to a live database at a custom IP and port, browse databases and
  collections, and download a collection by document index range or by UTC time range

Loaded files appear in the file list on the left. Any file can be removed from the session via its
delete button. Conversion and download status are logged in the panel's log output.

---

### Panel 2 — Monkey Data Extraction

A step-by-step wizard for downloading collections from a MongoDB instance:

1. Connect to the database (PCC network or external)
2. Select database and collection
3. Configure download range (by document index or time window)
4. Download and convert to CSV (with interpolation option)
5. Auto-load the converted file into the session

---

### Panel 3 — Data Processing

All transformations operate on in-memory copies of the data. **Changes are not written to disk until
you explicitly save the file using the save button in this panel.**

#### Operations

Select one or more columns and apply a single operation:

| Operation | Description |
|---|---|
| `normalize` | Min-max normalization of each selected column |
| `scale` | Multiply all values by a factor — parameter: `factor=x` |
| `offset` | Add a constant offset — parameter: `offset=x` |
| `flip_sign` | Negate all values |
| `absolute` | Take the absolute value of all values |
| `rename` | Rename a single selected column |
| `sort` | Sort the entire dataframe by a single column — parameter: `ascending=True/False` |
| `drop` | Remove data: by selected columns, by index range (e.g. `0,200`), or by a condition lambda (e.g. `rows["data.PT4"]>20`) |

#### Filters

Build a filter queue by adding filters in the desired execution order, then click **Apply** to run
the full queue on the selected columns.

| Filter | Parameters | Description |
|---|---|---|
| `low_pass` | `cutoff=x`, `order=x`, `fs=x` | Zero-phase Butterworth IIR low-pass filter |
| `adaptive_kalman` | `model=constant\|constant_velocity` | Adaptive Kalman filter |
| `wavelet_transform` | `wavelet_name=coif5`, `level=1–10`, `threshold_mode=soft\|hard` | Wavelet decomposition denoising |
| `rolling_mean` | `window=x` | Moving average over a sliding window |
| `rolling_median` | `window=x` | Moving median over a sliding window |
| `threshold` | `threshold=x` | Clip all values exceeding the threshold |
| `remove_negatives` | — | Replace all negative values with 0 |
| `remove_positives` | — | Replace all positive values with 0 |

Filtered columns are saved under a new name that encodes the filter chain, preserving the original
column in the dataset.

#### Frequency Analysis

Select a file and at least one column, then run either analysis. If the dataset contains a
recognizable time column (`header.timestamp_epoch`, `header.timestamp`, or `time`), the sampling
rate is estimated automatically; otherwise it defaults to 1.0 Hz or the manually entered value.
Results are saved automatically to the `plots/` directory as PNG files.

**FFT (Periodogram)**

| Parameter | Description |
|---|---|
| Fs [Hz] | Sampling frequency (auto-detected if left empty) |
| Samples for FFT | Number of samples used; larger values improve frequency resolution |
| Window | Windowing function: `hann`, `hamming`, `blackman`, `boxcar` |
| Detrend | Remove DC offset and linear trend before transform |
| dB scale | Display amplitude spectrum in decibels |
| Max freq [Hz] | Limit the displayed frequency axis |

**Spectrogram (STFT)**

| Parameter | Description |
|---|---|
| Fs [Hz] | Sampling frequency |
| Number of samples | Samples per STFT segment (default 512) |
| Overlap size | Overlapping samples between segments (default 50 % of segment length) |
| Window | `hann`, `hamming`, `blackman`, `boxcar` |
| Mode | `psd` (power spectral density), `magnitude`, `complex`, `angle`, `phase` |
| dB scale | Display values in decibels |
| Colormap | Matplotlib colormap name (e.g. `viridis`, `plasma`, `inferno`) |

---

### Panel 4 — Plotting

Visualize loaded and processed data. Plots are saved as PNG to the `plots/` directory.

#### Axes and series

- Add any loaded column to **Y1** (left axis) or **Y2** (right axis)
- Choose plot type per series: **Line** or **Scatter**
- Customize color, transparency, and marker size per series
- Set custom axis labels and plot title

#### X-axis

- Default X axis is `header.timestamp_epoch` (Unix timestamp in milliseconds)
- Convert to **seconds** or **milliseconds** relative to the first sample
- Apply a **time offset** (in ms) to shift a series left (negative) or right (positive) along the time axis — useful for aligning datasets manually

#### Reference lines

Add any number of horizontal or vertical dotted reference lines with custom colors to mark events,
ignition timing, or operating thresholds.

#### Dataset synchronization

When two datasets from different acquisition devices are loaded, the **Synchronize** button aligns
them automatically:

1. Detects the timestamp of the peak value in the selected column of each dataset
2. Computes the time delta between the two peaks
3. Applies an offset to dataset 2 so both peaks align with dataset 1
4. Reports the offset in milliseconds — paste it into the offset field to shift the plot to place
   ignition at t = 0 s

The synchronization overwrites the timestamp column in memory. Save the file to make it permanent.

---

## Data Format

Expected CSV structure after BSON conversion:

| Column | Description |
|---|---|
| `header.timestamp_epoch` | Unix timestamp in milliseconds |
| `header.timestamp_human` | Human-readable UTC timestamp |
| `header.counter` | Record sequence number |
| `header.origin` | Device / sensor identifier |
| `data.*` | Measurement channels (pressure, temperature, etc.) |

Supported device types in BSON parsing: LPB, ADV-USB, ADV-PCIE, COMP.

---

## Project Structure

```
postprocessing-app/
├── main.py                        # Application entry point
├── requirements.txt
│
├── gui/
│   ├── gui.py                     # Main window and panel navigation
│   └── panels/
│       ├── data_acquisition_panel.py
│       ├── monkey_panel.py
│       ├── data_processing_panel.py
│       └── plotting_panel.py
│
└── src/
    ├── data_acquisition.py        # BSON parsing and MongoDB client
    ├── data_processing.py         # DataFrame transformation API
    ├── filters.py                 # Signal filter implementations
    ├── plotter.py                 # Matplotlib rendering
    └── processing_utils.py        # Logging, threading utilities
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
