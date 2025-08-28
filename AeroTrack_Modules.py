import os
import pandas as pd
import numpy as np
from openap.phase import FlightPhase
from openap import Drag, prop
import matplotlib.pyplot as plt
import pandas as pd
from openap import prop
import traceback
from datetime import datetime
import numpy as np
from scipy.optimize import minimize
from openap import FuelFlow


def log_error(file_name, error_message, config_path=None, project_name=None):
    """
    Logs an error message to 'error.txt' in the same folder as the config.
    """
    try:
        error_dir = os.path.dirname(config_path) if config_path else os.getcwd()
        log_path = os.path.join(error_dir, (f"{project_name} error.txt"))
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.utcnow()} UTC] {file_name} → {error_message}\n")
    except Exception as fail:
        print(f"Failed to write error log: {fail}")


class FlightProcessor:
    def __init__(
        self, input_folder, output_folder, project_name, json_config_path, use_openap
    ):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.project_name = project_name.replace(" ", "")
        self.master_records = []
        self.json_config_path = json_config_path  # NEW
        self.use_openap = use_openap  # NEW

    def compute_vertical_rate(self, df):
        df["vertical_rate"] = df["Geoaltitude (ft)"].diff() / df["Time (s)"].diff() * 60
        df["vertical_rate"] = (
            df["vertical_rate"].fillna(0).rolling(window=5, min_periods=1).mean()
        )
        return df

    def compute_flight_path_angle(self, df):
        tas_fts = df["TAS (kn)"] * 1.68781
        delta_h = df["Geoaltitude (ft)"].diff()
        delta_t = df["Time (s)"].diff().replace(0, np.nan)
        vertical_component = delta_h / (tas_fts * delta_t)
        gamma_rad = np.arctan(vertical_component)
        df["flight_path_angle"] = gamma_rad.fillna(0)
        df["FPA (deg)"] = np.degrees(gamma_rad).fillna(0)
        return df

    def detect_basic_phases(self, df):
        phases = []
        window = 4
        alt_rolling = (
            df["Geoaltitude (ft)"].rolling(window=window, min_periods=1).mean()
        )

        for i in range(len(df)):
            alt = df.loc[i, "Geoaltitude (ft)"]
            vs = df.loc[i, "vertical_rate"]
            tas = df.loc[i, "TAS (kn)"]

            if alt < 1500 and tas > 50 and vs > 300:
                phase = "takeoff"
            elif vs > 300 and alt > 1500:
                phase = "climb"
            elif (
                i >= window
                and len(phases) > 0
                and phases[-1] == "cruise"
                and alt > alt_rolling[i] + 500
            ):
                phase = "step climb"
            elif (
                abs(vs) < 100
                and i > 5
                and abs(
                    df.loc[i, "Geoaltitude (ft)"] - df.loc[i - 5, "Geoaltitude (ft)"]
                )
                < 200
            ):
                phase = "cruise"
            elif vs < -300:
                phase = "descent"
            elif alt < 1000 and tas < 100:
                phase = "landing"
            else:
                phase = "unknown"

            phases.append(phase)

        df["basic_phase"] = phases
        return df

    def detect_flight_phase(self, df):
        time_sec = df["Time (s)"] - df["Time (s)"].iloc[0]
        alt_ft = df["Geoaltitude (ft)"].values
        spd_kts = df["TAS (kn)"].values
        roc_ftmin = df["vertical_rate"].values

        fp = FlightPhase()
        fp.set_trajectory(time_sec, alt_ft, spd_kts, roc_ftmin)
        df["flight_phase"] = fp.phaselabel()
        return df

    def process_file(
        self,
        file_name,
        index,
        total,
        lookup,
        aircraft_lookup,
        master_csv_path=None,
    ):
        if master_csv_path is None:
            master_csv_path = os.path.join(
                os.path.dirname(self.json_config_path),
                f"{self.project_name}_MasterFlight_List.csv",
            )
        base, _ = os.path.splitext(file_name)
        output_filename = base + "_phases.csv"
        output_path = os.path.join(self.output_folder, output_filename)

        if os.path.exists(output_path):
            print(f"[{index}/{total}] Skipping {file_name} (already processed)")
            return

        if not lookup.flight_exists(base):
            print(f"[{index}/{total}] Skipping {file_name} (not in master list)")
            return

        print(f"[{index}/{total}] Processing {file_name}")
        input_path = os.path.join(self.input_folder, file_name)
        df = pd.read_csv(input_path)

        df = self.compute_vertical_rate(df)
        df = self.compute_flight_path_angle(df)
        df = self.detect_flight_phase(df)
        df = self.detect_basic_phases(df)

        aircraft_data = aircraft_lookup.get_aircraft_properties(file_name)
        tow = "N/A"
        fuel_burnt = "N/A"
        time_spent = {
            p: 0
            for p in [
                "takeoff",
                "climb",
                "cruise",
                "step climb",
                "descent",
                "landing",
                "unknown",
            ]
        }

        if aircraft_data:
            typecode = str(aircraft_data.get("Typecode", "")).upper()

            try:
                fuel_estimator = FuelEstimator(aircraft_data)
                df, fuel_summary = fuel_estimator.estimate_fuel_burn(df)
                if "Fuel (kg)" in df.columns:
                    fuel_burnt = round(df["Fuel (kg)"].sum(), 2)
                print(f"  → Fuel estimate:\n{fuel_summary}")

                mission_estimator = MissionFuelAndWeightEstimator(df, aircraft_data)
                df = mission_estimator.compute_weight_over_time()
                if "Aircraft_Weight (kg)" in df.columns:
                    tow = round(df["Aircraft_Weight (kg)"].iloc[0], 2)
            except Exception as e:
                next
                print(f"Warning: Fuel estimation skipped → {e}")
                log_error(
                    file_name,
                    f"Weight Estimation Error: {e}",
                    self.json_config_path,
                    self.project_name,
                )

            try:
                drag_calc = DragCalculator(typecode, aircraft_data)
                df = drag_calc.compute_drag_and_weight(df)
            except Exception as e:
                print(f"Warning: Skipping drag calc for {file_name} → {e}")
            if self.use_openap == 1:
                try:
                    fuel_flow_estimator = FuelFlowEstimator(typecode)
                    df = fuel_flow_estimator.estimate_fuel_flow_series(df)
                    df = mission_estimator.compute_weight_openap_over_time()
                    fuel_burn_openap = (
                        df["Fuel_OpenAP (kg)"].sum()
                        if "Fuel_OpenAP (kg)" in df.columns
                        else np.nan
                    )
                    tow2_data = mission_estimator.estimate_tow_with_openap_fuel()

                except Exception as e:
                    print(f"Warning: OpenAP FuelFlow estimation failed → {e}")
                    log_error(
                        file_name,
                        f"FuelFlowEstimator error: {e}",
                        self.json_config_path,
                        self.project_name,
                    )
            try:
                if "Fuel (kg)" in df.columns:
                    engine_name = aircraft_data.get("engine", {}).get("default")
                    if engine_name:
                        engine_data = prop.engine(engine_name)
                        air_calc = AirMassFlowCalculator(engine_data)
                        df["air_mass_flow (kg/s)"] = air_calc.compute_air_mass_flow(
                            df["fuel_rate"].values
                        )
                        df["air_mass_flow_core (kg/s)"] = (
                            air_calc.compute_air_mass_flow_per_engine(
                                df["fuel_rate"].values
                            )
                        )
            except Exception as e:
                print(f"Warning: Air Mass Estimation skipped for {file_name} → {e}")
                log_error(
                    file_name,
                    f"Estimation error: {e}",
                    self.json_config_path,
                    self.project_name,
                )
            if self.use_openap == 1:
                try:
                    if "Fuel_OpenAP (kg)" in df.columns:
                        engine_name = aircraft_data.get("engine", {}).get("default")
                        if engine_name:
                            engine_data = prop.engine(engine_name)
                            air_calc_OPENAP = AirMassFlowCalculator(engine_data)
                            df["air_mass_flow_OPENAP (kg/s)"] = (
                                air_calc_OPENAP.compute_air_mass_flow(
                                    df["FuelFlow_OpenAP (kg/s)"].values
                                )
                            )
                            df["air_mass_flow_core_per_engine_OPENAP (kg/s)"] = (
                                air_calc_OPENAP.compute_air_mass_flow_per_engine(
                                    df["FuelFlow_OpenAP (kg/s)"].values
                                )
                            )
                            df["Dust Ingestion (kg)"] = (
                                df["dt"]
                                * df["Dust (large)"]
                                * df["air_mass_flow_core (kg/s)"]
                            )

                            df["Dust Ingestion OPENAP (kg)"] = (
                                df["dt"]
                                * df["Dust (large)"]
                                * df["air_mass_flow_core_per_engine_OPENAP (kg/s)"]
                            )

                except Exception as e:
                    print(
                        f"Warning: Air Mass Estimation (OPENAP) skipped for {file_name} → {e}"
                    )
                    log_error(
                        file_name,
                        f"Estimation error: {e}",
                        self.json_config_path,
                        self.project_name,
                    )

        if "basic_phase" in df.columns and "Time (s)" in df.columns:
            df["dt"] = df["Time (s)"].diff().fillna(0)
            phase_times = df.groupby("basic_phase")["dt"].sum().to_dict()
            for k in time_spent:
                if k in phase_times:
                    time_spent[k] = round(phase_times[k], 2)

        df.to_csv(output_path, index=False)
        if self.use_openap == 1:
            total_dust_OPENAP = {
                "Total Dust (g)": df["Dust Ingestion (kg)"].sum() * 1000,
                "Total Dust OPENAP (g)": df["Dust Ingestion OPENAP (kg)"].sum() * 1000,
            }
            master_record = {
                "Flight ID": base,
                "Engine": engine_name,
                "TOW (kg)": tow,
                "TOW2_OpenAP (kg)": tow2_data.get("TOW2", "N/A")
                if tow2_data
                else "N/A",
                "Fuel Burnt (kg) ": fuel_burnt,
                "Fuel Burnt OpenAP (kg)": fuel_burn_openap,
            }
            master_record.update(total_dust_OPENAP)
            master_record.update(time_spent)
            self.master_records.append(master_record)
        elif self.use_openap == 0:
            total_dust = {
                "Total Dust OPENAP (g)": df["Dust Ingestion (kg)"].sum() * 1000,
            }
            master_record = {
                "Flight ID": base,
                "Engine": engine_name,
                "TOW (kg)": tow,
                "Fuel Burnt (kg) ": fuel_burnt,
            }
            master_record.update(total_dust)
            master_record.update(time_spent)
            self.master_records.append(master_record)
        if master_csv_path:
            master_record = pd.DataFrame(self.master_records)
            master_record.to_csv(master_csv_path, index=False)

    def process_all_files(self, lookup, aircraft_lookup):
        csv_files = [f for f in os.listdir(self.input_folder) if f.endswith(".csv")]
        print(f"Found {len(csv_files)} flight files.")

        master_csv_path = os.path.join(
            os.path.dirname(self.output_folder),
            f"{self.project_name}_MasterFlight_List.csv",
        )

        for i, file_name in enumerate(csv_files, start=1):
            self.process_file(
                file_name, i, len(csv_files), lookup, aircraft_lookup, master_csv_path
            )


class FlightProcess:  ##Backup Class Incase of Failure
    def __init__(self, input_folder, output_folder):
        self.input_folder = input_folder
        self.output_folder = output_folder

    def compute_vertical_rate(self, df):
        df["vertical_rate"] = df["Geoaltitude (ft)"].diff() / df["Time (s)"].diff() * 60
        df["vertical_rate"] = (
            df["vertical_rate"].fillna(0).rolling(window=5, min_periods=1).mean()
        )
        return df

    def compute_flight_path_angle(self, df):
        tas_fts = df["TAS (kn)"] * 1.68781
        delta_h = df["Geoaltitude (ft)"].diff()
        delta_t = df["Time (s)"].diff().replace(0, np.nan)
        vertical_component = delta_h / (tas_fts * delta_t)
        gamma_rad = np.arctan(vertical_component)
        df["flight_path_angle"] = gamma_rad.fillna(0)
        df["FPA (deg)"] = np.degrees(gamma_rad).fillna(0)
        return df

    def detect_basic_phases(self, df):
        phases = []
        window = 5
        alt_rolling = (
            df["Geoaltitude (ft)"].rolling(window=window, min_periods=1).mean()
        )
        for i in range(len(df)):
            alt = df.loc[i, "Geoaltitude (ft)"]
            vs = df.loc[i, "vertical_rate"]
            tas = df.loc[i, "TAS (kn)"]

            # Safeguard for early rows
            if alt < 1500 and tas > 50 and vs > 300:
                phases = "takeoff"
            elif vs > 300 and alt > 1500:
                phases = "climb"
            elif (
                i >= window
                and len(phases) > 0
                and phases[-1] == "cruise"
                and alt > alt_rolling[i] + 500
            ):
                phases = "step climb"
            elif (
                abs(vs) < 100
                and i > 5
                and abs(
                    df.loc[i, "Geoaltitude (ft)"] - df.loc[i - 5, "Geoaltitude (ft)"]
                )
                < 200
            ):
                phases = "cruise"
            elif vs < -300:
                phases = "descent"
            elif alt < 1000 and tas < 100:
                phases = "landing"
            else:
                phases = "unknown"

            df["basic_phase"] = phases
            return df

    def detect_flight_phase(self, df):
        time_sec = df["Time (s)"] - df["Time (s)"].iloc[0]
        alt_ft = df["Geoaltitude (ft)"].values
        spd_kts = df["TAS (kn)"].values
        roc_ftmin = df["vertical_rate"].values

        fp = FlightPhase()
        fp.set_trajectory(time_sec, alt_ft, spd_kts, roc_ftmin)
        df["flight_phase"] = fp.phaselabel()
        return df

    def process_file(self, file_name, index, total, lookup, aircraft_lookup):
        base, _ = os.path.splitext(file_name)
        output_filename = base + "_phases.csv"
        output_path = os.path.join(self.output_folder, output_filename)

        if os.path.exists(output_path):
            print(f"[{index}/{total}] Skipping {file_name} (already processed)")
            return

        if not lookup.flight_exists(base):
            print(f"[{index}/{total}] Skipping {file_name} (not in master list)")
            return

        print(f"[{index}/{total}] Processing {file_name}")
        input_path = os.path.join(self.input_folder, file_name)
        df = pd.read_csv(input_path)

        df = self.compute_vertical_rate(df)
        df = self.compute_flight_path_angle(df)
        df = self.detect_flight_phase(df)
        df = self.detect_basic_phases(df)  # This is our version
        # Lookup aircraft properties
        aircraft_data = aircraft_lookup.get_aircraft_properties(file_name)

        # Try to compute drag only if MTOW is available

        if aircraft_data:
            typecode = str(aircraft_data.get("Typecode", "")).upper()

            try:
                fuel_estimator = FuelEstimator(aircraft_data)
                df, fuel_summary = fuel_estimator.estimate_fuel_burn(df)
                # print(f"  → Fuel estimate:\n{fuel_summary}")
                print("  → Fuel estimate:\n")
                for count in range(len(fuel_summary)):
                    print(f" \n{fuel_summary[count]}")
                mission_estimator = MissionFuelAndWeightEstimator(df, aircraft_data)
                df = mission_estimator.compute_weight_over_time()

            except Exception as e:
                print(f"Warning: Fuel estimation skipped → {e}")
                log_error(
                    file_name,
                    f"Fuel estimation error: {e}",
                    self.json_config_path,
                    self.project_name,
                )

            try:
                drag_calc = DragCalculator(typecode, aircraft_data)
                df = drag_calc.compute_drag_and_weight(df)
            except Exception as e:
                print(f"Warning: Skipping drag calc for {file_name} → {e}")
                log_error(
                    file_name,
                    f"Drag calculation error: {e}",
                    self.json_config_path,
                    self.project_name,
                )

            try:
                if "total_drag" in df.columns:
                    estimator = MassFlowEstimator(df["total_drag"].values)
                    df["mass_flow_rate"] = estimator.estimate()
                    A, B = estimator.get_parameters()
                    print(f"  → Mass flow model: Drag = m × {A:.2f} + {B:.2f}")
                else:
                    print("  → Skipping mass flow estimation: 'drag' column missing.")
            except Exception as e:
                print(f"Warning: Skipping mass flow estimation for {file_name} → {e}")
                log_error(
                    file_name,
                    f"Mass flow estimation error: {e}",
                    self.json_config_path,
                    self.project_name,
                )

            # Save results regardless
        df.to_csv(output_path, index=False)

    def process_all_files(self, lookup, aircraft_lookup):
        csv_files = [f for f in os.listdir(self.input_folder) if f.endswith(".csv")]
        print(f"Found {len(csv_files)} flight files.")
s
        for i, file_name in enumerate(csv_files, start=1):
            self.process_file(file_name, i, len(csv_files), lookup, aircraft_lookup)


class MasterFlightLookup:
    def __init__(self, lookup_folder):
        self.lookup_folder = lookup_folder
        self.master_df = self.load_master_list()

    def load_master_list(self):
        all_csvs = [f for f in os.listdir(self.lookup_folder) if f.endswith("MASTER.csv")]
        if not all_csvs:
            raise FileNotFoundError(
                "No CSV files found in the master flight list folder."
            )

        dfs = []
        for file in all_csvs:
            df = pd.read_csv(os.path.join(self.lookup_folder, file))
            dfs.append(df)

        master_df = pd.concat(dfs, ignore_index=True)
        master_df["Flight_ID"] = master_df["Flight_ID"].astype(str).str.strip()
        return master_df

    def flight_exists(self, filename_no_ext):
        flight_id = os.path.splitext(filename_no_ext)[0]
        return flight_id in self.master_df["Flight_ID"].values

    def find_typecode_for_flight(self, filename_with_or_without_csv):
        flight_id = os.path.splitext(filename_with_or_without_csv)[0]
        match = self.master_df[self.master_df["Flight_ID"] == flight_id]

        if not match.empty:
            typecode = match.iloc[0]["Typecode"]
            print(f"  Flight '{flight_id}' → Typecode: {typecode}")
        else:
            print(f"  Flight '{flight_id}' not found in master list.")
            typecode = False
        return typecode


class AircraftPropertiesLookup:
    def __init__(self, master_lookup):
        self.master_lookup = master_lookup

    def get_aircraft_properties(self, flight_id):
        filename_no_ext = os.path.splitext(flight_id)[0]
        match = self.master_lookup.master_df[
            self.master_lookup.master_df["Flight_ID"] == filename_no_ext
        ]

        if match.empty:
            print(f"Skipping: '{filename_no_ext}' not found in master flight list.")
            return None

        typecode = str(match.iloc[0]["Typecode"]).strip().upper()

        try:
            aircraft_data = prop.aircraft(typecode)
            aircraft_data["Typecode"] = typecode
            print(f"Aircraft: {filename_no_ext} → Typecode: '{typecode}'")
            return aircraft_data
        except Exception as e:
            print(f"Error fetching aircraft data for typecode '{typecode}': {e}")
            return None


class DragCalculator:
    def __init__(self, typecode, aircraft_properties):
        """
        Uses OpenAP Drag module to estimate clean configuration drag
        using actual time-varying aircraft weight.
        """
        self.typecode = str(typecode).upper()
        self.drag_model = Drag(ac=self.typecode)

    def compute_drag_and_weight(self, df):
        # Check that required columns exist
        try:
            required_columns = [
                "Geoaltitude (ft)",
                "TAS (kn)",
                "vertical_rate",
                "FPA (deg)",
                "Aircraft_Weight (kg)",
            ]
        except:
            required_columns = [
                "Geoaltitude (ft)",
                "TAS (kn)",
                "vertical_rate",
                "FPA (deg)",
                "Aircraft_Weight_OpenAP (kg)",
            ]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in DataFrame.")

        alt_ft = df["Geoaltitude (ft)"].values
        tas_kn = df["TAS (kn)"].values
        vs_fpm = df["vertical_rate"].values
        weight_kg = df["Aircraft_Weight (kg)"].values
        gamma_rad = np.radians(df["FPA (deg)"].values)

        drag_vals = []
        weight_components = []

        for alt, tas, vs, angle, wt in zip(
            alt_ft, tas_kn, vs_fpm, gamma_rad, weight_kg
        ):
            try:
                D = self.drag_model.clean(mass=wt, tas=tas, alt=alt, vs=vs)
                drag_vals.append(D)
            except Exception:
                drag_vals.append(np.nan)

            weight_component = wt * 9.81 * np.sin(angle)
            weight_components.append(weight_component)

        df["total_drag"] = drag_vals
        df["weight_component (N)"] = weight_components
        return df


class Plotting:
    def __init__(self, output_folder):
        self.output_folder = output_folder

    def plot_drag_vs_time(self):
        files = [f for f in os.listdir(self.output_folder) if f.endswith("_phases.csv")]
        if not files:
            print("No _phases.csv files found in output folder.")
            return

        print("\nAvailable files:")
        for idx, file in enumerate(files):
            print(f"[{idx}] {file}")

        try:
            choice = int(input("\nEnter file number to plot: "))
            selected_file = files[choice]
        except (ValueError, IndexError):
            print("Invalid selection.")
            return

        df = pd.read_csv(os.path.join(self.output_folder, selected_file))

        if "total_drag" not in df.columns or "weight_component" not in df.columns:
            print("Selected file does not contain drag data.")
            return

        plt.figure(figsize=(12, 6))
        plt.plot(df["Time (s)"], df["total_drag"], label="Total Drag (N)")
        plt.plot(df["Time (s)"], df["weight_component"], label="Weight Component (N)")
        plt.xlabel("Time (s)")
        plt.ylabel("Force (N)")
        plt.title(f"Drag and Weight Component vs Time\n{selected_file}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


class FuelFlowEstimator:  # This is the OpenAP Fuel Flow Estimator
    def __init__(self, typecode):
        self.typecode = typecode
        self.model = FuelFlow(ac=typecode)

    def estimate_enroute(self, mass, tas, alt, vs=None):
        try:
            return self.model.enroute(mass=mass, tas=tas, alt=alt, vs=vs)
        except Exception:
            return np.nan

    def estimate_fuel_flow_series(self, df):
        required_cols = {
            "TAS (kn)",
            "Geoaltitude (ft)",
            "vertical_rate",
            "Aircraft_Weight (kg)",
            "basic_phase",
        }
        if not required_cols.issubset(df.columns):
            raise ValueError("Required columns are missing from DataFrame.")

        tas_vals = df["TAS (kn)"].values
        alt_vals = df["Geoaltitude (ft)"].values
        vs_vals = df["vertical_rate"].values
        mass_vals = df["Aircraft_Weight (kg)"].values
        phases = df["basic_phase"].values

        fuel_flows = []
        for tas, alt, vs, mass, phase in zip(
            tas_vals, alt_vals, vs_vals, mass_vals, phases
        ):
            try:
                if phase == "takeoff":
                    ff = self.model.takeoff(tas=tas, alt=alt, throttle=1.0)
                elif phase == "cruise":
                    ff = self.model.enroute(mass=mass, tas=tas, alt=alt)
                elif phase == "climb" or phase == "step climb":
                    ff = self.model.enroute(mass=mass, tas=tas, alt=alt, vs=vs)
                elif phase == "descent" or phase == "landing":
                    ff = self.model.enroute(mass=mass, tas=tas, alt=alt, vs=vs)
                else:
                    ff = self.model.enroute(mass=mass, tas=tas, alt=alt)
            except Exception:
                ff = np.nan

            fuel_flows.append(ff)

        df["FuelFlow_OpenAP (kg/s)"] = fuel_flows
        df["FuelFlow_OpenAP (kg/s)"] = df["FuelFlow_OpenAP (kg/s)"].clip(lower=0)
        df["Fuel_OpenAP (kg)"] = df["FuelFlow_OpenAP (kg/s)"] * df["dt"]
        # Calculate and print total fuel burned per phase
        try:
            phase_fuel_totals = df.groupby("basic_phase")["Fuel_OpenAP (kg)"].sum()
            print("\nOpenAP Fuel Burn by Phase:")
            for phase, total in phase_fuel_totals.items():
                print(f"  {phase}: {total:.2f} kg")
        except Exception as e:
            print(f"Error printing fuel burn per phase: {e}")

        return df


class FuelEstimator:  ##This is the steady State Fuel Estimator
    def __init__(self, aircraft_data):
        engine_name = aircraft_data.get("engine", {}).get("default", None)
        if not engine_name:
            raise ValueError("Engine data not found in aircraft data.")

        self.engine_params = prop.engine(engine_name)
        self.num_engines = aircraft_data.get("engine", {}).get("number", 2)

        self.phase_rate_map = {
            "takeoff": self.engine_params.get("ff_to", 1.0),
            "climb": self.engine_params.get("ff_co", 0.9),
            "cruise": self.engine_params.get("ff_cr", 0.8),  # fallback key
            "step climb": self.engine_params.get("ff_co", 0.9),
            "descent": self.engine_params.get("ff_idl", 0.1),
            "landing": self.engine_params.get("ff_app", 0.3),
            "unknown": self.engine_params.get("ff_idl", 0.1),
        }

    def estimate_fuel_burn(self, df):
        if "basic_phase" not in df.columns or "Time (s)" not in df.columns:
            raise ValueError(
                "DataFrame must include 'basic_phase' and 'Time (s)' columns"
            )

        # Compute time delta between each row (seconds)
        df["dt"] = df["Time (s)"].diff().fillna(0)

        # Compute fuel burn rate (kg/s) based on phase
        df["fuel_rate"] = (
            df["basic_phase"].apply(
                lambda x: self.phase_rate_map.get(str(x).lower(), 0.0)
            )
            * self.num_engines
        )

        # Fuel burned per timestep
        df["Fuel (kg)"] = df["fuel_rate"] * df["dt"]
        # Total fuel burn per phase
        phase_totals = df.groupby("basic_phase")["Fuel (kg)"].sum().to_dict()
        # df["Total_Fuel_Burn (kg)"] = np.sum(df["fuel_rate"] * df["dt"])
        return df, phase_totals


class MissionFuelAndWeightEstimator:
    def __init__(self, df, aircraft_data):
        self.df = df
        self.aircraft_data = aircraft_data
        self.mtow_kg = aircraft_data.get("limits", {}).get("MTOW")
        self.oew_kg = aircraft_data.get("limits", {}).get("OEW")
        self.mfc_kg = aircraft_data.get("limits", {}).get("MFC")
        self.total_fuel_used = df["Fuel (kg)"].sum() if "Fuel (kg)" in df else None
        self.initial_weight = None  # Will be set after estimating weight
        self.segment_fraction = None
        self.fuel_weight_fraction = None
        self.skip_weight_calc = False

    def compute_segment_weight_fractions(self):
        if "Fuel (kg)" not in self.df.columns:
            raise ValueError("Fuel data not found in DataFrame")

        grouped = self.df.groupby("basic_phase")["Fuel (kg)"].sum()
        total_fuel = grouped.sum()
        self.segment_fraction = (grouped / total_fuel).to_dict()
        return self.segment_fraction

    def compute_mission_fuel_fraction(self):
        if self.total_fuel_used is None or not self.mtow_kg:
            raise ValueError("Fuel usage or MTOW not available.")

        self.fuel_weight_fraction = self.total_fuel_used / self.mtow_kg
        return self.fuel_weight_fraction

    def estimate_weight_iteratively(self):
        pax_data = self.aircraft_data.get("pax", {})
        pax_avg = round((pax_data.get("low") + pax_data.get("high")) / 2)
        pax_weight = 80  # kg per passenger including seat and life support
        bag_weight = 20  # average kg per bag
        bags = pax_avg  # assume 1 bag per pax
        cargo = 0  # optional to define

        w_payload = pax_avg * pax_weight + bags * bag_weight + cargo
        fuel_weight = self.total_fuel_used if self.total_fuel_used else 0

        if self.oew_kg is None:
            raise ValueError("OEW (Operating Empty Weight) is not available.")

        self.initial_weight = (self.oew_kg + w_payload + fuel_weight) * 1

        if self.initial_weight < self.oew_kg:
            raise ValueError(
                f"Estimated TOW ({self.initial_weight:.1f} kg) is below OEW ({self.oew_kg:.1f} kg). Check fuel and payload assumptions."
            )

        while self.mtow_kg and self.initial_weight > self.mtow_kg:
            print(
                f"Warning: Estimated TOW ({self.initial_weight:.1f} kg) exceeds MTOW ({self.mtow_kg:.1f} kg). Adjusting Payload weight ."
            )
            w_payload = w_payload - (self.initial_weight - self.mtow_kg)
            self.initial_weight = self.oew_kg + w_payload + fuel_weight
            # self.skip_weight_calc = False

        empty_weight = self.initial_weight - (fuel_weight + w_payload)

        self.print_summary_table()

        return {
            "initial_weight": round(self.initial_weight, 2),
            "fuel_weight": round(fuel_weight, 2),
            "payload_weight": round(w_payload, 2),
            "empty_weight": round(empty_weight, 2),
        }

    def estimate_weight_breakdown(self):
        return self.estimate_weight_iteratively()

    def compute_weight_over_time(self):
        if self.initial_weight is None:
            weight_data = self.estimate_weight_iteratively()
            if weight_data is None:
                self.df["Aircraft_Weight (kg)"] = ""
                return self.df

        self.df["Aircraft_Weight (kg)"] = (
            self.initial_weight - self.df["Fuel (kg)"].cumsum()
        )
        return self.df

    def print_summary_table(self):
        if self.segment_fraction is None:
            self.compute_segment_weight_fractions()
        if self.fuel_weight_fraction is None:
            self.compute_mission_fuel_fraction()

        print("\nTakeOff Weight:")
        print(f"  TOW (kg): {self.initial_weight:.4f}")

    def estimate_tow_with_openap_fuel(self):
        if "Fuel_OpenAP (kg)" not in self.df.columns:
            raise ValueError("Fuel_OpenAP (kg) not found in DataFrame.")

        openap_fuel = self.df["Fuel_OpenAP (kg)"].sum()

        pax_data = self.aircraft_data.get("pax", {})
        pax_avg = round((pax_data.get("low", 150) + pax_data.get("high", 170)) / 2)
        pax_weight = 80
        bag_weight = 20
        bags = pax_avg
        cargo = 0

        w_payload = pax_avg * pax_weight + bags * bag_weight + cargo

        if self.oew_kg is None:
            raise ValueError("OEW (Operating Empty Weight) is not available.")

        tow2 = (self.oew_kg + w_payload + openap_fuel) * 1
        print("\nEstimated TOW with OpenAP Fuel:")
        print(f"  TOW2 (kg): {tow2:.4f}")
        return {
            "TOW2": round(tow2, 2),
            "OpenAP_Fuel": round(openap_fuel, 2),
            "Payload": round(w_payload, 2),
            "OEW": round(self.oew_kg, 2),
        }

    def compute_weight_openap_over_time(self):
        if "Fuel_OpenAP (kg)" not in self.df.columns:
            raise ValueError("Fuel_OpenAP (kg) column not found in DataFrame.")

        try:
            tow_data = self.estimate_tow_with_openap_fuel()
            initial_weight = tow_data.get("TOW2", None)
            if initial_weight is None:
                raise ValueError("TOW2 could not be computed from OpenAP fuel data.")
            self.df["Aircraft_Weight_OpenAP (kg)"] = (
                initial_weight - self.df["Fuel_OpenAP (kg)"].cumsum()
            )
            return self.df
        except Exception as e:
            print(f"Error computing weight from OpenAP fuel: {e}")
            return self.df


class AirMassFlowCalculator:
    def __init__(self, engine_data, afr=60):
        self.bpr = engine_data.get("bpr", None)
        self.afr = afr
        if self.bpr is None:
            raise ValueError("Bypass Ratio (bpr) not found in engine data.")

    def compute_air_mass_flow(self, fuel_mass_flow):
        fuel_mass_flow = np.asarray(fuel_mass_flow)
        m_air = self.afr * fuel_mass_flow * (1 + self.bpr)
        # m_air_per_engine = 0.5 * m_air * (self.bpr / (1 + self.bpr))
        return m_air

    def compute_air_mass_flow_per_engine(self, fuel_mass_flow):
        fuel_mass_flow = np.asarray(fuel_mass_flow)
        m_air = self.afr * fuel_mass_flow * (1 + self.bpr)
        # m_air_per_engine = 0.5 * m_air * (self.bpr / (1 + self.bpr))
        m_air_per_engine = m_air / (1 + self.bpr)
        return m_air_per_engine


class MassFlowEstimator:
    def __init__(self, drag_values):
        """
        Initialize the estimator using known drag values (equal to thrust at steady state).
        :param drag_values: np.array of drag [N] per timestep
        """
        self.drag = np.array(drag_values)
        self.n = len(drag_values)
        self.mass_flow = None
        self.A = None
        self.B = None

    def _loss(self, x):
        """
        Loss function to minimize: sum of squared errors between estimated drag and actual drag.
        x[:n] = estimated mass flow (m_i) at each timestep
        x[-2] = A, x[-1] = B
        """
        m = x[: self.n]
        A = x[-2]
        B = x[-1]
        model_drag = m * A + B
        return np.sum((self.drag - model_drag) ** 2)

    def estimate(self):
        """
        Estimate mass flow at each timestep, and the linear model parameters A and B.
        """
        # Initial guesses: m = drag / 1000, A = 1000, B = 0
        m0 = self.drag / 1000
        A0 = 100
        B0 = 0
        x0 = np.concatenate([m0, [A0, B0]])

        # Bounds to ensure physical plausibility: m > 0, A > 0
        bounds = [(1e-6, None)] * self.n + [(1e-6, None), (None, None)]

        result = minimize(self._loss, x0, bounds=bounds, method="L-BFGS-B")

        if result.success:
            x_opt = result.x
            self.mass_flow = x_opt[: self.n]
            self.A = x_opt[-2]
            self.B = x_opt[-1]
            return self.mass_flow
        else:
            raise RuntimeError(f"Mass flow estimation failed: {result.message}")

    def get_parameters(self):
        """
        Return the fitted values for A and B in the model Drag = m*A + B.
        """
        return self.A, self.B


class MiscUtilities:
    @staticmethod
    def get_isa_conditions(altitude_m):
        T0 = 288.15  # K
        P0 = 101325  # Pa
        rho0 = 1.225  # kg/m^3
        g = 9.80665  # m/s^2
        R = 287.058  # J/(kg·K)
        L = -0.0065  # K/m

        if altitude_m <= 11000:
            T = T0 + L * altitude_m
            P = P0 * (T / T0) ** (-g / (L * R))
            rho = rho0 * (T / T0) ** (-g / (L * R) - 1)
        else:
            T = 216.65
            h_tropopause = 11000
            T_tropopause = T0 + L * h_tropopause
            P_tropopause = P0 * (T_tropopause / T0) ** (-g / (L * R))
            delta_h = altitude_m - h_tropopause
            P = P_tropopause * np.exp(-g * delta_h / (R * T))
            rho = P / (R * T)

        return P, rho, T


class MasterFileGenerator:
    def __init__(self, json_path, processed_folder):
        self.json_path = json_path
        self.processed_folder = processed_folder
        self.master_data = []

    def collect_flight_data(self):
        csv_files = [
            f for f in os.listdir(self.processed_folder) if f.endswith("_phases.csv")
        ]

        for file in csv_files:
            try:
                flight_id = file.replace("_phases.csv", "")
                df = pd.read_csv(os.path.join(self.processed_folder, file))

                tow = (
                    df["Aircraft_Weight (kg)"].iloc[0]
                    if "Aircraft_Weight (kg)" in df.columns
                    else None
                )
                fuel_burnt = (
                    df["Fuel (kg)"].sum() if "Fuel (kg)" in df.columns else None
                )

                time_spent = (
                    df.groupby("basic_phase")["dt"].sum().to_dict()
                    if "dt" in df.columns
                    else {}
                )
                time_spent_str = {k: round(v, 1) for k, v in time_spent.items()}

                self.master_data.append(
                    {
                        "Flight ID": flight_id,
                        "TOW": round(tow, 1) if tow else None,
                        "Fuel Burnt": round(fuel_burnt, 1) if fuel_burnt else None,
                        **time_spent_str,
                    }
                )

            except Exception as e:
                print(f"Error processing {file}: {e}")

    def generate_master_file(self):
        output_name = input(
            "Enter name for master summary CSV file (without extension): "
        ).strip()
        output_csv = os.path.join(
            os.path.dirname(self.json_path),
            output_name + "Valid_MasterFlight_List" + ".csv",
        )

        self.collect_flight_data()

        if not self.master_data:
            print("No flight data collected.")
            return

        df_master = pd.DataFrame(self.master_data)
        df_master.to_csv(output_csv, index=False)
        print(f"Master summary written to: {output_csv}")
