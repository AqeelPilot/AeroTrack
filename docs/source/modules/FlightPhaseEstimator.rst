FlightPhaseEstimator
====================

The `FlightPhaseEstimator` class identifies distinct flight phases—such as climb, cruise, and descent—based on altitude and vertical speed profiles. It is a core component in AeroTrack for categorizing aircraft behavior post-flight and isolating segments for targeted analysis (e.g., cruise-only fuel burn).

Class Overview
--------------

.. autoclass:: AeroTrack_Modules.FlightPhaseEstimator
   :members:
   :undoc-members:
   :show-inheritance:

Constructor
-----------

.. code-block:: python

   FlightPhaseEstimator()

Initializes the estimator without requiring additional parameters. Logic is internally configured to assign flight phases based on vertical speed and smoothed barometric altitude.

Key Methods
-----------

.. autosummary::
   :toctree: generated/

   AeroTrack_Modules.FlightPhaseEstimator.estimate_phases
   AeroTrack_Modules.FlightPhaseEstimator.identify_cruise_segments

**Method Descriptions:**

- **`estimate_phases(df)`**  
  Adds a `phase` column to the DataFrame, labeling each time point as `"Climb"`, `"Cruise"`, `"Descent"`, or `"Other"` based on vertical speed thresholds.

- **`identify_cruise_segments(df)`**  
  Returns a filtered DataFrame containing only the cruise segments, useful for subsequent cruise-only analysis (e.g., CO-temperature correlation, drag comparisons).

Flight Phase Logic
------------------

The classification is based on:

- **Climb**: vertical speed > +500 ft/min  
- **Descent**: vertical speed < -500 ft/min  
- **Cruise**: vertical speed between ±500 ft/min AND stable altitude  
- **Other**: Ground or transitional phases

Usage Example
-------------

.. code-block:: python

   from AeroTrack_Modules import FlightPhaseEstimator

   estimator = FlightPhaseEstimator()
   df = estimator.estimate_phases(df)

   cruise_df = estimator.identify_cruise_segments(df)

Output
------

The modified DataFrame includes:

- `phase`: Flight phase classification
- Optionally, only cruise segments when using `identify_cruise_segments`

Visualization Tip
-----------------

Plotting `baro_alt_AC` against time colored by phase helps visually validate the classification:

.. code-block:: python

   import matplotlib.pyplot as plt

   phases = {"Climb": "orange", "Cruise": "green", "Descent": "blue", "Other": "gray"}
   for phase, color in phases.items():
       plt.plot(df[df["phase"] == phase]["UTC_time"],
                df[df["phase"] == phase]["baro_alt_AC"],
                label=phase, color=color)

   plt.legend()
   plt.xlabel("Time")
   plt.ylabel("Altitude (m)")
   plt.title("Flight Phases")
   plt.show()

Dependencies
------------

- `numpy`
- `pandas`
