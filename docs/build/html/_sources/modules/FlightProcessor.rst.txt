FlightProcessor
===============

The `FlightProcessor` class is the core engine of AeroTrack. It manages the end-to-end post-processing of aircraft flight trajectory CSV files, integrating configuration files, master flight lookup data, and OpenAP performance models.

This class is responsible for:
 - Loading individual flight data files
 - Estimating key physical parameters (drag, fuel burn, altitude profile)
 - Identifying flight phases (climb, cruise, descent)
 - Outputting processed data to structured CSVs for further analysis

Class Overview
--------------

.. autoclass:: AeroTrack_Modules.FlightProcessor
   :members:
   :undoc-members:
   :show-inheritance:

Constructor
-----------

.. code-block:: python

   FlightProcessor(input_path, output_path, project_name, config_path, use_openap)

**Parameters:**

- `input_path` (str): Folder containing raw flight CSV files.
- `output_path` (str): Folder where processed files will be saved.
- `project_name` (str): Name identifier for the current run.
- `config_path` (str): Path to the `project_config.json` file.
- `use_openap` (int): Flag to toggle OpenAP-based dynamic modeling.

Key Methods
-----------

.. autosummary::
   :toctree: generated/

   AeroTrack_Modules.FlightProcessor.process_file
   AeroTrack_Modules.FlightProcessor.calculate_vertical_speed_and_fpa
   AeroTrack_Modules.FlightProcessor.estimate_drag_components
   AeroTrack_Modules.FlightProcessor.estimate_fuel_burn
   AeroTrack_Modules.FlightProcessor.estimate_weight_component
   AeroTrack_Modules.FlightProcessor.identify_flight_phases
   AeroTrack_Modules.FlightProcessor.flag_cruise_segments
   AeroTrack_Modules.FlightProcessor.apply_aircraft_mass

Usage Example
-------------

.. code-block:: python

   from AeroTrack_Modules import FlightProcessor

   processor = FlightProcessor(
       input_path="FlightData/",
       output_path="Processed/",
       project_name="QatarFleet",
       config_path="project_config.json",
       use_openap=1
   )

   processor.process_file("20230718_A6EVO_A350.csv", index=1, total=10, flight_lookup, aircraft_lookup)

Output
------

Processed CSVs contain the following enriched columns:

- `flight_path_angle`, `vertical_speed`
- `total_drag`, `weight_component`
- `fuel_burn`, `estimated_mass`
- `flight_phase`, `cruise_flag`

These values provide a high-fidelity aerodynamic and operational profile for each flight, enabling further analytics and visualization within the AeroTrack or DustFlight ecosystem.

Dependencies
------------

- `pandas`
- `numpy`
- `math`
- `openap`
- `tqdm` (for progress bar)

