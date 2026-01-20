RocketDAQ Analyzer
---
The developed application constitutes an integral part of the engineering thesis entitled
“Digital filtering of data from pressure sensors in the rocket engine system.” </br>
The software was created to support the analysis and post-processing of experimental data 
acquired during rocket engine test campaigns and serves as a practical implementation of the methods discussed in this work.
---

## Features
- BSON data retrieval
- BSON to JSON conversion
- JSON to csv conversion
- Data cleanup and analysis
- Plotting of the csv data and saving them as separate files
- Plotting of the flight telemetry on animated plots

---

## Getting Started



### Prerequisites

Ensure you have the following installed:

- Python 3.10+
- `pip` (Python package installer)

---

### Setting Up the Environment

1. **Clone the Repository**

   Clone the repository to your local machine:

   ```bash
   git clone https://github.com/AndrewJHK/RocketDAQ-Analyzer
   cd RocketDAQ-Analyzer
   ```

2. **Create a Virtual Environment**

   It's recommended to use a virtual environment to manage dependencies. Run the following command to create a virtual
   environment:

   ```bash
   python -m venv venv
   ```

3. **Activate the Virtual Environment**

   Activate the virtual environment using the appropriate command for your operating system:

    - **Windows:**

      ```bash
      venv\Scripts\activate
      ```

    - **macOS and Linux:**

      ```bash
      source venv/bin/activate
      ```

4. **Install Required Packages**

   With the virtual environment activated, install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

App consists of 4 main panels

- Data acquisition
- Flight plot
- Data processing
- Plotting

---

### Data acquisition

Here you can retrieve data straight from the MongoDB database as BSON types based on two possible indexes:

- **Document number** - Selecting the appropriate button and filling out the start and stop indexes will result in retrieval
    of that range of documents - rows.
- **Date** - Selecting the time related button and filling out the start and end date will result in retrieval
    of all the documents - rows - that have been saved in provided time frame.

On top of that this panel supports the loading the csv data, conversion of json files to csv and conversion of bson files to json. Loaded files will show up on the left in the list with an
adjacent delete button.</br>
JSON loading has two radio buttons:

- **Interpolate** - Every column will have a value for every timestamp that will appear.
  Sometimes for a specific timestamp only 3 of 6 channels sent data. In that case value from a previous timestamp shall
  be assigned to this one.
- **Fill None** - When there is no value for specific column in a specific timestamp it will be assigned 'None'.

After conversion of JSON file you still need to load them as csv files.


---

### Flight Plot

Choose a csv file that contains required data columns - header.timestamp_epoch, data.telemetry.acc_data.\*,
data.telemetry.quaternion.\*
Simply click compute button and wait for the results - computed apogee, max speed and plots. Currently, the animation
for
rotation is not working so its disabled and waiting for fix.
If you want to save the computed data remember to click the save to file button.

---

### Data Processing

This panel is responsible for all kinds of data transformation, analysis and filtration. </br> </br>
**REMEMBER THAT ANY DATA CHANGES THAT CAN BE MADE IN THIS PANEL ARE NOT REFLECTED IN THE FILE ITSELF. THEY WILL BE IF
YOU
SAVE SAID DATA INTO A FILE WITH A PROVIDED BUTTON.**</br></br>
When you select a data that you want to transform, all the available columns will show up with a checkbox for an easy
selection.

---

#### Operations

Possible operations are:

- **normalize** - take content of each selected column and perform a min-max value normalization of them
- **scale** - take content of each selected column and scale them by a factor provided in parameters box - 'factor=x'
- **flip_sign** - take content of each selected columns and change the sign + into -, </br> - into +
- **sort** - select only one column and sort the whole data by that specific column. In parameters specify if it should
  be
- **rename** - select only one column and rename it
  ascending or descending by writing 'ascending=True/False'
- **drop** - when selecting drop operation, three radio buttons will appear:
    - **Drop by columns** - the selected columns will be deleted
    - **Drop by index range** - all rows in the range provided in params will be deleted fe. 0,200
    - **Drop by condition (lambda)** - all rows that meet the specified condition provided in parameters will be deleted
      fe 'rows["data.PT4.scaled"]>20'

---

#### Filters

Firstly queue all the filters in an order that you wish they should be executed for a specified columns. Then hit the
apply button to start the application queue.</br>
Possible filters are:

- **remove_negatives** - replace all negative values with 0
- **remove_positives** - replace all positive values with 0
- **rolling_mean** - perform a rolling mean filter with a specified windows size in parameters fe. 'window=10'
- **rolling_median** - perform a rolling median filter with a specified windows size in parameters fe. 'window=10'
- **threshold** - replace all the values that exceed a provided value with that value fe. 'threshold=35'
- **wavelet_transform** - perform a wavelet decomposition and recomposition with specified parameters fe. '
  wavelet_name=coif5,level=10,threshold_mode=soft'.
    - **wavelet_name** - specify what kind of wavelet decomposition to use. After testing 'coif5' seems to be working
      the
      best. List in [documentation](https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html)
    - **level** - level of decomposition and smoothing of the signal to perform, </br> range of integer values 1-10. The
      higher the level the smoother the signal which means that details might be lost
    - **threshold_mode** - type of thresholding either soft or hard. Soft provides a smoother result.

---

#### Frequency Analysis

This section allows you to perform **frequency-domain analysis** on any numeric signal present in your data.  
It provides two main tools: **Fast Fourier Transform (FFT)** and **Spectrogram** computation.

After selecting a file and at least one column (signal) from the list, you can configure and plot either an FFT or a spectrogram.  
If your dataset contains a column representing time (for example `header.timestamp_epoch`, `header.timestamp`, or `time`), it will be **automatically detected and used** for frequency estimation.  
If no such column is found, the analysis assumes a sampling rate of `1.0 Hz` or uses the value provided manually.

---

##### FFT (Fast Fourier Transform)

This tool generates a **frequency spectrum** of the selected signal, showing how the signal’s amplitude or power is distributed across frequencies.

**Parameters:**
- **Fs [Hz]** — sampling frequency.  
  Leave empty to automatically infer from the detected time column (if present).  
  If no time column is available, a default value of `1.0 Hz` is used.
- **Samples for FFT** — number of samples used for the FFT computation.  
  Larger values provide better frequency resolution at the cost of longer computation time.
- **Window** — type of windowing function applied before the transform (`hann`, `hamming`, `blackman`, `boxcar`).
- **Detrend** — if enabled, removes the DC component and any linear trend before transformation.
- **dB scale** — if enabled, displays the amplitude spectrum in decibels.
- **Max freq [Hz]** — optionally limit the displayed frequency range.
- **Plot FFT** — computes and displays the FFT of the selected signal.

The FFT result is automatically saved in the `plots/` directory as a `.png` file.

---

##### Spectrogram

The spectrogram visualizes how the frequency content of a signal changes over time, using a short-time Fourier transform (STFT).

**Parameters:**
- **Fs [Hz]** — sampling frequency, inferred automatically if a recognizable time column exists.
- **Number of samples** — number of samples per STFT segment (default: `512`).
- **Overlap size** — number of overlapping samples between segments (default: 50% of `nperseg` if left empty).
- **Window** — window function used for each segment (`hann`, `hamming`, `blackman`, `boxcar`).
- **Mode** — determines what is shown in the spectrogram (`psd`, `magnitude`, `complex`, `angle`, `phase`).
  - **`psd`** *(Power Spectral Density)* — shows signal power per frequency band, proportional to energy content.  
    This is the default and most commonly used mode for physical signals such as pressure or voltage.
  - **`magnitude`** — displays the absolute magnitude of the FFT for each time window, without squaring.  
    Useful when you care about amplitude, not power.
  - **`complex`** — shows complex FFT coefficients (real + imaginary).  
    Mostly used for debugging or when you need phase-sensitive information.
  - **`angle`** — visualizes the phase angle (argument) of each FFT component in radians.  
    Helpful for phase tracking between frequencies.
  - **`phase`** — similar to `angle`, but wraps the phase to the range `[-π, π]`.  
    This is useful for analyzing phase shifts or synchronization between signals.
- **dB scale** — whether to display values in decibels.
- **Colormap** — name of the color palette used for visualization (for example: `viridis`, `plasma`, `inferno`).
- **Plot Spectrogram** — generates and displays the spectrogram for the selected signal.

The resulting spectrogram image is also saved automatically in the `plots/` directory.

---

### Plotting

#### General info

Here you can plot data that you loaded and transformed in the previous panel. It is possible to have two separately
scaled Y axes and combining a plot from two different databases. The X axis can be changed, by default the X axis is
'header.timestamp_epoch' but it can be changed for each data column individually. </br>
On top of that it can be automatically
converted to seconds and milliseconds, as well shift the
graph in time by providing the offset in milliseconds in the specified box. Negative values shift the graph to the left
and positive to the right.
On the bottom you can enter horizontal and vertical dotted lines.

#### Synchronization

The provided button will try to synchronize timestamps of two databases. </br>
Synchronization is tuned for static engine tests, it will look for the highest values of selected columns, and compare
their timestamps.
After that it will synch db2 to db1 and provide an offset in milliseconds to copy and paste into offset window resulting
in shifting the whole graph so the ignition happens at 0 seconds. The synchronization will overwrite the data, but as
always it needs to be saved to be preserved.

### Deactivating the Virtual Environment

When you're done working on the project, deactivate the virtual environment by running:

```bash
deactivate
```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
