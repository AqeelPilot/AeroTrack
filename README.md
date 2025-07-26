# ✈️ Flight Data Analysis and Drag Estimation using OpenAP

This project automates the analysis of commercial aircraft flight data with the aim of estimating total drag and evaluating flight phases using aerodynamic and engine parameters from the [OpenAP](https://openap.aero) library. It integrates a master flight program list, aircraft specifications, and dynamically computes key performance metrics including:

* Vertical speed and flight path angle
* Flight phase identification (e.g., climb, cruise)
* Total drag (clean configuration)
* Weight component during ascent
* Fuel flow estimation using OpenAP engine models

## 📁 Project Structure

```
.
├── Flight_Program_Main.py                # Main script to process flight data
├── Flight_Program_Lookup_Modules.py     # Module for MTOW lookup and drag calculation
├── aircraft_data/                        # JSON files with aircraft performance details
├── flight_data/                          # Folder with input CSV files for each flight
├── output_data/                          # Folder where processed CSV files are saved
├── master_flight_list.csv               # Lookup table containing flight program metadata
└── README.md                             # This file
```

## 🧠 Features

* Calculates flight dynamics such as:

  * Vertical speed from altitude and time
  * Flight path angle from TAS and vertical speed
* Computes drag using OpenAP for clean configuration
* Supports aircraft-specific lookups via typecode and MTOW
* Handles unknown aircraft gracefully (logs and skips drag computation)
* Generates enriched CSVs with appended columns for:

  * `vertical_speed_ft_min`
  * `flight_path_angle_deg`
  * `flight_phase`
  * `total_drag_N` (if MTOW and typecode found)
  * `weight_component_N` (same condition as above)

## ⚙️ Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/flight-drag-analysis.git
cd flight-drag-analysis
```

### 2. Install Dependencies

```bash
pip install pandas numpy openap
```

### 3. Organize Your Data

* Place raw flight CSV files in `flight_data/`
* Ensure `master_flight_list.csv` exists and contains:

  * `data_file_name` (without `.csv` extension)
  * `typecode`, etc.

### 4. Run the Script

```bash
python Flight_Program_Main.py
```

Processed files will be saved in the `output_data/` folder.

## 🧹 Module Descriptions

### `Flight_Program_Main.py`

* Loads the master list
* Iterates over all available flight files
* Applies calculations and exports results

### `Flight_Program_Lookup_Modules.py`

* `lookup_mtow_and_typecode()`: Fetches MTOW and typecode from the master list
* `calculate_vertical_speed()`: Computes vertical speed in ft/min
* `calculate_flight_path_angle()`: Computes flight path angle in degrees
* `calculate_total_drag_and_weight_component()`: Uses OpenAP to estimate drag and weight force
* Handles missing aircraft data with warnings and skips drag computation when needed

## 🔍 Example Output

Sample output CSV will have:

| timestamp | altitude\_m | tas\_knots | vertical\_speed\_ft\_min | flight\_path\_angle\_deg | flight\_phase | total\_drag\_N | weight\_component\_N |
| --------- | ----------- | ---------- | ------------------------ | ------------------------ | ------------- | -------------- | -------------------- |
| ...       | ...         | ...        | ...                      | ...                      | ...           | ...            | ...                  |

## 📌 Notes

* Drag estimation assumes **clean configuration only**.
* If aircraft data is missing from the OpenAP database, only the basic calculations are done.
* Ensure all filenames in the master list match the actual files (excluding `.csv`).

