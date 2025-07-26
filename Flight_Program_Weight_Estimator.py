import numpy as np
import math


class AircraftWeightEstimator:
    def __init__(self, aircraft):
        self.aircraft = aircraft
        self.Wo = [5500.0]  # Initial guess for takeoff weight (kg)
        self.iteration_limit = 5000

    def get_atmos_properties(self, altitude_m):
        # Simplified ISA model
        T0 = 288.15
        lapse_rate = -0.0065
        T = T0 + lapse_rate * altitude_m
        return T

    def convergence(self, W_payload, FuelWeightFraction, Wo_prev, error, k):
        empty_weight_fraction = 0.97 * Wo_prev**-0.06
        denominator = 1 - FuelWeightFraction - empty_weight_fraction
        if denominator <= 0 or not np.isreal(denominator):
            Wo_new = (1 + error) ** 2 * Wo_prev
        else:
            Wo_new = W_payload / denominator
        new_error = abs((Wo_new - Wo_prev) / Wo_new)
        return denominator, new_error, Wo_new

    def estimate(self):
        ac = self.aircraft

        # Payload weight
        W_payload = (
            ac["Pax"] * ac["PaxWeight"] + ac["Bags"] * ac["BagWeight"] + ac["Cargo"]
        )
        W_payload = round(W_payload, 2)

        # Atmosphere
        T = self.get_atmos_properties(ac["CruiseAlt"] * 0.3048)
        V = ac["CruiseMach"] * math.sqrt(1.4 * 287.1 * T)

        # Aerodynamics
        k = 1 / (math.pi * ac["e"] * ac["AR"])
        CL_opt_range = math.sqrt(ac["CD0"] / (3 * k))
        CD = ac["CD0"] + k * CL_opt_range**2
        ClSAR = CL_opt_range / CD
        ratioLD = 0.8667 / math.sqrt(4 * ac["CD0"] * k)

        # Cruise segment
        TSFC = ac["c_cruise"] / 3600
        range_m = (ac["Range"] - ac["ClimbDescentCredit"]) * 1852
        CruiseSeg = math.exp(-range_m * TSFC / (V * ratioLD))

        # Loiter
        loiter_time = 30 * 60  # seconds
        EF = ratioLD / TSFC
        LoiterSegFrac = math.exp(-loiter_time / EF)

        # Alternate cruise
        altRange = (200 - 100) * 1852
        altCruiseSegFrac = math.exp(-altRange * TSFC / (V * ratioLD))

        # Segment fractions
        seg = [
            0.995,  # WarmTaxi
            0.99,  # TakeOff
            0.98,  # Climb
            CruiseSeg,
            0.995,  # Descent
            0.995,  # Landing
            LoiterSegFrac,
            0.999,  # Taxi
            0.99,  # altClimb
            altCruiseSegFrac,
            0.995,  # altLanding
        ]

        # Compute mission fractions
        missionFraction = [seg[0]]
        for s in seg[1:]:
            missionFraction.append(missionFraction[-1] * s)
        seg.append(1 - 0.0476)  # Reserve fuel segment
        missionFraction.append(missionFraction[-1] * seg[-1])
        FuelWeightFraction = 1 - missionFraction[-1]

        # Iteration loop
        error = 0.1
        z = 0
        while error > 0.01 and z < self.iteration_limit:
            denom, error, Wo_new = self.convergence(
                W_payload, FuelWeightFraction, self.Wo[-1], error, z
            )
            self.Wo.append(Wo_new)
            z += 1

        Initial_Weight = round(self.Wo[-1], 2)
        FuelWeight = round(FuelWeightFraction * Initial_Weight, 2)
        Empty_Weight = round(Initial_Weight - (FuelWeight + W_payload), 2)

        return {
            "ClSAR": ClSAR,
            "Initial_Weight": Initial_Weight,
            "FuelWeight": FuelWeight,
            "MaxPayload": ac["MaxPyld"],
            "Payload_Weight": W_payload,
            "Empty_Weight": Empty_Weight,
            "SegmentFraction": seg,
            "MissionFraction": missionFraction,
        }
