Modules
=======

This section provides a high-level overview of the main modules that make up the AeroTrack pipeline. Each module is designed to handle a specific aspect of the flight analysis process, from reading data to estimating performance and exporting results.

AeroTrack is divided into two primary files:

- `AeroTrack_MAIN.py`: The entry point and runtime controller
- `AeroTrack_Modules.py`: Core module definitions and logic

Module Overview
---------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Module
     - Description
   * - `AeroTrack_MAIN`
     - Manages the full AeroTrack execution workflow, from reading the project configuration to saving processed results. Prompts user input via GUI and runs the full pipeline on all CSVs in the specified input folder.
   * - `FlightProcessor`
     - Core class that handles all CSV file-level processing. Responsible for calculating vertical speed, flight path angle, identifying flight phases, estimating drag, fuel burn, and saving enhanced outputs.
   * - `FlightPhaseEstimator`
     - Uses OpenAP's `FlightPhase` model to assign phases (Takeoff, Climb, Cruise, etc.) to each row of flight data based on speed, altitude, and time history.
   * - `DragEstimator`
     - Computes total aerodynamic drag and weight component (sine of flight path angle) using OpenAP's clean drag models and estimated mass.
   * - `FuelBurnEstimator`
     - Estimates per-second and cumulative fuel burn using either a fixed lookup table or OpenAP's engine-specific fuel flow model.
   * - `DustIngestionEstimator`
     - (Optional) Estimates air mass flow through the engine and integrates with external dust concentration data to compute dust ingestion levels during cruise.
   * - `Plotter`
     - Generates optional visualizations including drag curves, fuel burn profiles, and flight phase transitions.
   * - `Utilities`
     - A collection of helper functions for time conversion, angle math, file management, and user prompts.

Modularity
----------

Each class is designed to work independently and can be imported in custom scripts or notebooks if needed. The `FlightProcessor` acts as the orchestrator, coordinating all calculations.

Module Reference
----------------

For detailed class and method documentation, refer to the following:

- :doc:`modules/FlightProcessor`
- :doc:`modules/DragEstimator`
- :doc:`modules/FuelBurnEstimator`
- :doc:`modules/FlightPhaseEstimator`
- :doc:`modules/DustIngestionEstimator`
- :doc:`modules/Plotter`

