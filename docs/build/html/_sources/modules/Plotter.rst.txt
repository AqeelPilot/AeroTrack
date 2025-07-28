
Plotter
=======

The `Plotter` class provides visualization tools for analyzing flight data processed by AeroTrack. It generates time-series and correlation plots for key parameters like CO concentration, air temperature, fuel burn, and altitude, offering visual insights into flight behavior and environmental conditions.

Class Overview
--------------

.. autoclass:: AeroTrack_Modules.Plotter
   :members:
   :undoc-members:
   :show-inheritance:

Constructor
-----------

.. code-block:: python

   Plotter()

Initializes the plotting utility. No external parameters are needed—methods directly take in processed DataFrames.

Key Methods
-----------

.. autosummary::
   :toctree: generated/

   AeroTrack_Modules.Plotter.plot_co_temperature_correlation
   AeroTrack_Modules.Plotter.plot_fuel_burn_and_altitude
   AeroTrack_Modules.Plotter.plot_dust_ingestion_profile
   AeroTrack_Modules.Plotter.plot_flight_phases

Method Descriptions
-------------------

- **`plot_co_temperature_correlation(df)`**  
  Plots CO concentration vs. air temperature for the cruise segment of a flight. Helpful for detecting wildfire smoke signatures.

- **`plot_fuel_burn_and_altitude(df)`**  
  Generates a dual-axis plot of fuel burn and barometric altitude over time to visualize aircraft power demand during different flight phases.

- **`plot_dust_ingestion_profile(df)`**  
  Creates a scatter plot of barometric altitude vs. dust ingestion mass to assess vertical distribution of airborne dust exposure.

- **`plot_flight_phases(df)`**  
  Uses color-coded segmentation of the altitude profile to show Climb, Cruise, Descent, and Other phases.

Usage Example
-------------

.. code-block:: python

   from AeroTrack_Modules import Plotter

   plotter = Plotter()
   plotter.plot_co_temperature_correlation(df)
   plotter.plot_fuel_burn_and_altitude(df)
   plotter.plot_dust_ingestion_profile(df)
   plotter.plot_flight_phases(df)

Output
------

All plots are rendered using `matplotlib.pyplot` and displayed interactively (or saved if needed). These visuals support detailed post-flight analysis and comparison between environmental exposure and engine behavior.

Plot Example
------------

Here's an example showing a CO-temperature correlation in cruise:

.. code-block:: python

   plotter.plot_co_temperature_correlation(df)

.. image:: ../_static/sample_plot_co_temp.png
   :alt: CO vs Temperature
   :width: 600px

Dependencies
------------

- `matplotlib`
- `pandas`
- `numpy`

