Usage
=======

This section explains how to run AeroTrack, interpret its inputs and outputs, and understand its processing workflow.

Running AeroTrack
-----------------

After installation, launch the main script:

.. code-block:: bash

    python AeroTrack_MAIN.py

A file dialog will open prompting you to select a configuration file (`project_config.json`). This file defines key paths for input and output data.

Configuration File Example
--------------------------

The configuration file must be a JSON file with the following structure:

.. code-block:: json

    {
      "input_folder": "FlightCSVs/",
      "output_folder": "Processed/",
      "master_flight_list_folder": "MasterList/",
      "project_name": "SampleProject"
    }

- `input_folder`: Path to a folder containing raw flight `.csv` files
- `output_folder`: Where processed outputs will be saved
- `master_flight_list_folder`: Folder containing the aircraft metadata file
- `project_name`: Used as a prefix in output filenames

Expected Folder Structure
-------------------------

.. code-block:: text

    AeroTrack/
    ├── FlightCSVs/
    │   ├── 20230701_1210_A320_XXXXX.csv
    │   └── ...
    ├── Processed/
    ├── MasterList/
    │   └── Master_Flight_List.csv
    ├── project_config.json
    ├── AeroTrack_MAIN.py
    └── AeroTrack_Modules.py

Processing Steps
----------------

Once a valid configuration is selected, AeroTrack performs the following:

1. **Loads each `.csv` file** in the input folder  
2. **Extracts physical quantities**: vertical speed, flight path angle, etc.  
3. **Identifies flight phases** using OpenAP's `FlightPhase` model  
4. **Estimates drag** using OpenAP's aerodynamic models (if aircraft type known)  
5. **Calculates fuel burn** using either fixed lookup or OpenAP's engine model  
6. **Saves processed outputs** to the output folder:
   - Annotated flight file (e.g., `20230701_1210_A320_XXXXX_phases.csv`)
   - Optional plots (drag, fuel, phase time series)
7. **Updates a master summary file** with metrics across all flights

Optional Prompts
----------------

At runtime, AeroTrack will ask whether to:

- **Enable drag and weight estimation** via OpenAP
- **Enable fuel burn estimation**
- **Enable plotting** of key performance variables

These allow users to control the level of computation performed.

Output Files
------------

Each processed flight produces:

- A CSV file with new columns:
  - `Vertical_Speed`
  - `Flight_Path_Angle`
  - `Flight_Phase`
  - `Total_Drag` (if OpenAP is used)
  - `Fuel_Burn` (if enabled)
- A master `.csv` file (e.g., `SampleProject_MasterFlightList.csv`) containing a summary row per flight

Visualizations
--------------

If enabled, AeroTrack generates:
- **Time-series plots**: drag vs time, fuel vs time
- **Flight phase transitions**
- **Fuel burn vs altitude or temperature**

These are saved to the output folder or displayed interactively.

Next Step
---------

To learn how each module contributes to the AeroTrack pipeline, continue to the :doc:`modules` section or explore the :doc:`api_reference/FlightProcessor` page.
