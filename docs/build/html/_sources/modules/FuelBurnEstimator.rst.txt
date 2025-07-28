FuelBurnEstimator
=================

The `FuelBurnEstimator` class in AeroTrack estimates fuel consumption based on aircraft engine performance characteristics, thrust, and flight dynamics. It retrieves engine-specific data from the OpenAP library and applies a physics-informed estimation model to compute fuel burn rates.

This module plays a key role in post-flight performance analysis and fuel efficiency studies.

Class Overview
--------------

.. autoclass:: AeroTrack_Modules.FuelBurnEstimator
   :members:
   :undoc-members:
   :show-inheritance:

Constructor
-----------

.. code-block:: python

   FuelBurnEstimator(typecode)

**Parameters:**

- `typecode` (str): ICAO aircraft type designator (e.g., `'A320'`, `'B744'`).

This initializes the estimator by retrieving the default engine model and associated performance parameters (e.g., fuel flow coefficients) from OpenAP.

Key Methods
-----------

.. autosummary::
   :toctree: generated/

   AeroTrack_Modules.FuelBurnEstimator.estimate_fuel_burn

**Method Descriptions:**

- **`estimate_fuel_burn(df)`**  
  Estimates fuel consumption (in kg/s) using modeled thrust and specific fuel consumption (SFC) for each timestamp in the DataFrame. Adds a `fuel_burn_kg_s` column.

Usage Example
-------------

.. code-block:: python

   from AeroTrack_Modules import FuelBurnEstimator

   estimator = FuelBurnEstimator("A320")
   df = estimator.estimate_fuel_burn(df)

   df.to_csv("flight_with_fuel.csv", index=False)

Output
------

The output DataFrame will include:

- `fuel_burn_kg_s`: Estimated fuel flow rate in kilograms per second at each time step.

Estimation Logic
----------------

Fuel burn is computed using:

.. math::

   \text{Fuel Burn (kg/s)} = \text{SFC} \times \text{Thrust (N)}

Where:
- SFC is derived from OpenAP engine models.
- Thrust is typically estimated based on vertical speed, drag, and climb/descent phase.

Dependencies
------------

- `openap`
- `numpy`
- `pandas`

