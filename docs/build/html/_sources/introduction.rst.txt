Introduction
============

**AeroTrack** is a modular post-flight analysis toolkit designed to extract aerodynamic and performance metrics from aircraft trajectory data. It was developed by **Muhammad Aqeel Abdulla** as part of a broader research initiative into aircraft behavior under varying environmental and operational conditions.

Originally written to support the **Dust Flight Dashboard**, AeroTrack provides the computational foundation for large-scale flight diagnostics, combining raw flight CSVs with OpenAP models and empirical estimation routines. It is built for researchers, analysts, and engineers who need to quantify fuel consumption, drag, flight phase distribution, and more.

Author
------

This tool was developed by:

**Muhammad Aqeel Abdulla**  
PhD Student, Aerospace Engineering  
The University of Manchester  
[aqeelabdulla.me](https://aqeelabdulla.me)

Key Capabilities
----------------

AeroTrack performs the following functions automatically:

- **Flight Dynamics Computation**  
  Calculates vertical speed, flight path angle, and flight phase labeling using OpenAP's `FlightPhase` module.

- **Aerodynamic Drag & Weight Breakdown**  
  Estimates total drag and weight-induced components using time-resolved OpenAP clean configuration models.

- **Fuel Burn Estimation**  
  Supports both steady-state engine mappings and OpenAP's dynamic fuel flow models, segmented by flight phase.

- **Air Mass Flow & Dust Ingestion (Optional)**  
  If engine and dust concentration data are available, it can compute core air mass flow and cumulative dust mass ingested.

- **CSV Output & Plotting**  
  Enhances each input `.csv` with new physical columns and optionally produces summary plots.

Use Cases
---------

AeroTrack has been used in:

- **Dust ingestion research** integrating MERRA-2 and METAR datasets
- **Fuel efficiency studies** across various aircraft and engine types
- **Environmental exposure analysis** at cruise altitudes
- **Large-scale processing** of thousands of flights with automated summaries

Documentation Structure
------------------------

This documentation will guide you through:

- Installing AeroTrack and its dependencies
- Setting up the required folder structure and JSON configuration
- Understanding the modular design of processors, estimators, and utilities
- Generating and interpreting enriched outputs

AeroTrack is actively maintained and modular by design, making it easy to adapt to new datasets, models, or operational goals.
