# postprocessing-app

postprocessing-app is a Python-based postprocessing application aimed at the problem of readability of data acquired
during tests

## Features

- JSON to csv conversion
- Data cleanup and analysis
- Plotting of the csv data and saving them as separate files
- Plotting of the flight telemetry on animated plots

## Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.10+
- `pip` (Python package installer)

### Setting Up the Environment

1. **Clone the Repository**

   Clone the repository to your local machine:

   ```bash
   git clone https://github.com/AndrewJHK/postprocessing-app
   cd postprocessing-app
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

## Usage

App consists of 4 main panels

- Load Data
- Flight Plot
- Data processing
- Plotting

### Load Data

Here you can load the csv data and convert json files to csv. Loaded files will show up on the left in the list with an
adjacent delete
button.</br>
JSON loading has two radio buttons:

- **Interpolate** - Every column will have a value for every timestamp that will appear.
  Sometimes for a specific timestamp only 3 of 6 channels sent data. In that case value from a previous timestamp shall
  be assigned to this one.
- **Fill None** - When there is no value for specific column in a specific timestamp it will be assigned 'None'.

After conversion of JSON file you still need to load them as csv files.

### Flight Plot

Choose a csv file that contains required data columns - header.timestamp_epoch, data.telemetry.acc_data.\*,
data.telemetry.quaternion.\*
Simply click compute button and wait for the results - computed apogee, max speed and plots. Currently, the animation
for
rotation is not working so its disabled and waiting for fix.
If you want to save the computed data remember to click the save to file button.

### Data Processing

This panel is responsible for all kinds of data transformation and filtration. </br> </br>
**REMEMBER THAT ANY DATA CHANGES THAT CAN BE MADE IN THIS PANEL ARE NOT REFLECTED IN THE FILE ITSELF. THEY WILL BE IF
YOU
SAVE SAID DATA INTO A FILE WITH A PROVIDED BUTTON.**</br></br>
When you select a data that you want to transform, all the available columns will show up with a checkbox for an easy
selection.

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
