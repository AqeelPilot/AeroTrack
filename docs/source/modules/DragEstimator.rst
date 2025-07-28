DragEstimator
=============

The `DragEstimator` class in AeroTrack calculates the total drag experienced by an aircraft throughout its flight using aerodynamic models and flight parameters. It integrates with the OpenAP library to dynamically determine drag characteristics based on aircraft type, speed, altitude, and flight path angle.

This module is essential for evaluating aircraft performance and efficiency post-flight.

Class Overview
--------------

.. autoclass:: AeroTrack_Modules.DragEstimator
   :members:
   :undoc-members:
   :show-inheritance:

Constructor
-----------

.. code-block:: python

   DragEstimator(typecode)

**Parameters:**

- `typecode` (str): ICAO aircraft type designator (e.g., `'A332'`, `'B77W'`).

This initializes the aircraft model from OpenAP and stores aircraft-specific parameters such as maximum takeoff weight (MTOW).

Key Methods
-----------

.. autosummary::
   :toctree: generated/

   AeroTrack_Modules.DragEstimator.get_mtow
   AeroTrack_Modules.DragEstimator.estimate_total_drag

**Method Descriptions:**

- **`get_mtow()`**  
  Returns the aircraft’s maximum takeoff weight (MTOW) in kilograms using OpenAP.

- **`estimate_total_drag(df, vertical_speed, tas, altitude)`**  
  Calculates total drag (in Newtons) for each row in a flight DataFrame based on current speed, altitude, and climb angle. Adds `total_drag` as a new column to `df`.

Usage Example
-------------

.. code-block:: python

   from AeroTrack_Modules import DragEstimator

   estimator = DragEstimator("A332")
   mtow = estimator.get_mtow()

   df = estimator.estimate_total_drag(
       df,
       vertical_speed=df["vertical_speed"],
       tas=df["tas_knots"],
       altitude=df["altitude_m"]
   )

   df.to_csv("flight_with_drag.csv", index=False)

Output
------

The output DataFrame includes an additional column:

- `total_drag`: Estimated drag force in Newtons, computed using OpenAP aerodynamic models.

Dependencies
------------

- `openap`
- `numpy`
- `pandas`

