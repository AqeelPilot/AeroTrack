DustIngestionEstimator
======================

The `DustIngestionEstimator` class quantifies the potential exposure of an aircraft to dust during flight by interpolating dust concentration data along the flight path. This module is designed to interface with datasets like MERRA-2 or other gridded environmental sources, making it a key component for environmental impact analysis in AeroTrack.

Class Overview
--------------

.. autoclass:: AeroTrack_Modules.DustIngestionEstimator
   :members:
   :undoc-members:
   :show-inheritance:

Constructor
-----------

.. code-block:: python

   DustIngestionEstimator(dust_data)

**Parameters:**

- `dust_data` (dict): A dictionary containing gridded dust concentration values, typically in the format `{(lat, lon, alt): dust_mass}` or gridded arrays with matching metadata.

The class stores and prepares the dust map for fast interpolation.

Key Methods
-----------

.. autosummary::
   :toctree: generated/

   AeroTrack_Modules.DustIngestionEstimator.estimate_ingestion

**Method Descriptions:**

- **`estimate_ingestion(df)`**  
  Interpolates dust concentration values for each point in the flight path based on latitude, longitude, and barometric altitude. Appends a new column `dust_ingestion` to the DataFrame, which can be used for downstream analysis or plotting.

Dust Interpolation Logic
-------------------------

The ingestion is estimated by:

- Mapping each flight point to the nearest grid cells
- Optionally applying trilinear or nearest-neighbor interpolation
- Assigning a mass loading value in µg/m³ or kg/m³, depending on source data

This allows regional or temporal spikes in airborne dust to be matched with the aircraft’s exact trajectory.

Usage Example
-------------

.. code-block:: python

   from AeroTrack_Modules import DustIngestionEstimator

   dust_data = load_merra2_csv("20220315_DUSMASS.csv")  # Example external utility
   estimator = DustIngestionEstimator(dust_data)

   df = estimator.estimate_ingestion(df)
   df.to_csv("flight_with_dust.csv", index=False)

Output
------

- `dust_ingestion`: Estimated surface mass concentration (e.g., µg/m³) experienced by the aircraft at each flight point.

Visualization Tip
-----------------

To visualize dust levels along the altitude profile:

.. code-block:: python

   import matplotlib.pyplot as plt

   plt.scatter(df["baro_alt_AC"], df["dust_ingestion"], c="brown", s=5)
   plt.xlabel("Barometric Altitude (m)")
   plt.ylabel("Dust Mass Concentration (µg/m³)")
   plt.title("Dust Ingestion Profile")
   plt.grid(True)
   plt.show()

Dependencies
------------

- `numpy`
- `pandas`
- `scipy` (for interpolation)
