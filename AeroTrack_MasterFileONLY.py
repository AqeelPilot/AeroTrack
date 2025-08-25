# AeroTrack_MasterFileONLY_patched.py
# - Adds a new input `masterflightlist` (file OR folder of CSVs)
# - Looks up aircraft engine using OpenAP (same protocol as in Modules)
# - Adds "Engine" as Column 2 (right after "Flight ID") in the output MasterFlightList
# - Keeps existing metrics and behavior

import os
import pandas as pd
import time
import datetime

try:
    # Use the same OpenAP protocol as AeroTrack_Modules
    from openap import prop
except Exception as e:
    prop = None
    print(f"Warning: OpenAP not available → {e}")


def _read_master_csvs(masterflightlist):
    """
    Reads one or more master flight list CSVs and returns a normalized DataFrame.
    Accepts either a directory (reads all CSVs) or a single CSV file.
    Expects columns including 'Flight_ID' and 'Typecode' (case-insensitive tolerant).
    """
    import os
    import pandas as pd

    path = str(masterflightlist)
    if os.path.isdir(path):
        csvs = [
            os.path.join(path, f)
            for f in os.listdir(path)
            if f.lower().endswith(".csv")
        ]
        if not csvs:
            raise FileNotFoundError(
                "No CSV files found in the masterflightlist folder."
            )
        dfs = [pd.read_csv(p) for p in csvs]
        df = pd.concat(dfs, ignore_index=True)
    elif os.path.isfile(path):
        df = pd.read_csv(path)
    else:
        raise FileNotFoundError(f"masterflightlist not found: {path}")

    # Normalize column names
    cols = {c.lower(): c for c in df.columns}
    # Try common variants
    flight_col = cols.get("flight_id", None) or cols.get("flight id", None)
    typecode_col = cols.get("typecode", None)

    if not flight_col or not typecode_col:
        raise KeyError(
            "Master flight list must include 'Flight_ID' and 'Typecode' columns (case-insensitive)."
        )

    df = df.rename(columns={flight_col: "Flight_ID", typecode_col: "Typecode"})
    df["Flight_ID"] = df["Flight_ID"].astype(str).str.strip()
    df["Typecode"] = df["Typecode"].astype(str).str.strip().str.upper()
    return df


def _build_engine_lookup(masterflightlist_df):
    """
    Builds a lookup dict: Flight_ID -> Engine name (variant) using OpenAP.
    If OpenAP is not available or lookup fails, returns 'N/A' for that flight.
    """
    engine_map = {}
    if prop is None:
        # OpenAP not available; return all N/A
        for fid in masterflightlist_df["Flight_ID"].unique():
            engine_map[fid] = "N/A"
        return engine_map

    # Cache per-typecode to avoid repeated OpenAP calls
    typecode_to_engine = {}
    for _, row in masterflightlist_df.iterrows():
        fid = row["Flight_ID"]
        tcode = row["Typecode"]

        if tcode not in typecode_to_engine:
            try:
                ac = prop.aircraft(tcode)
                engine_name = (ac.get("engine") or {}).get("default", "N/A")
            except Exception:
                engine_name = "N/A"
            typecode_to_engine[tcode] = engine_name

        engine_map[fid] = typecode_to_engine.get(tcode, "N/A")
    return engine_map


def generate_master_flight_list(phases_folder, masterflightlist, output_path):
    """
    Build the MasterFlightList by scanning *_phases.csv files, pulling summary stats,
    and (NEW) inserting the Engine variant (from OpenAP via masterflightlist Typecode)
    as Column 2 after 'Flight ID'.

    Parameters
    ----------
    phases_folder : str
        Folder containing per-flight '*_phases.csv' files.
    masterflightlist : str
        Path to a single CSV OR a directory of CSVs that together form the master list.
        Must include columns: Flight_ID, Typecode.
    output_path : str
        Destination CSV path for the consolidated MasterFlightList.
    """
    start = time.time()
    master_records = []

    # NEW: Read the masterflightlist and build engine map
    mdf = _read_master_csvs(masterflightlist)
    engine_map = _build_engine_lookup(mdf)

    # Collect phase files
    csv_files = [f for f in os.listdir(phases_folder) if f.endswith("_phases.csv")]
    print(f"Found {len(csv_files)} phase files.")

    for idx, file in enumerate(csv_files, start=1):
        print(f"[{idx}/{len(csv_files)}] Processing {file}")
        try:
            file_path = os.path.join(phases_folder, file)
            df = pd.read_csv(file_path)

            flight_id = file.replace("_phases.csv", "")
            engine_name = engine_map.get(flight_id, "N/A")
            print(f"  - Flight ID: {flight_id}, Engine: {engine_name}")
            record = {
                # Ensure Engine sits in Column 2 by constructing the dict in order
                "Flight ID": flight_id,
                "Engine": engine_name,  # NEW: Engine variant directly after Flight ID
                "TOW (kg)": round(df["Aircraft_Weight (kg)"].iloc[0], 2)
                if "Aircraft_Weight (kg)" in df.columns
                else "N/A",
                "TOW2_OpenAP (kg)": round(df["Aircraft_Weight_OpenAP (kg)"].iloc[0], 2)
                if "Aircraft_Weight_OpenAP (kg)" in df.columns
                else "N/A",
                "Fuel Burnt (kg) ": round(df["Fuel (kg)"].sum(), 2)
                if "Fuel (kg)" in df.columns
                else "N/A",
                "Fuel Burnt OpenAP (kg)": round(df["Fuel_OpenAP (kg)"].sum(), 2)
                if "Fuel_OpenAP (kg)" in df.columns
                else "N/A",
                "Total Dust Ingested (g)": (df["Dust Ingestion (kg)"].sum() * 1000)
                if "Dust Ingestion (kg)" in df.columns
                else "N/A",
                "Total Dust Ingested OPENAP (g)": (
                    df["Dust Ingestion OPENAP (kg)"].sum() * 1000
                )
                if "Dust Ingestion OPENAP (kg)" in df.columns
                else "N/A",
            }

            # Phase time aggregation
            if "dt" in df.columns and "basic_phase" in df.columns:
                phase_times = df.groupby("basic_phase")["dt"].sum().to_dict()
                for phase in [
                    "takeoff",
                    "climb",
                    "cruise",
                    "step climb",
                    "descent",
                    "landing",
                    "unknown",
                ]:
                    record[phase] = round(phase_times.get(phase, 0), 1)

            master_records.append(record)

        except Exception as e:
            print(f"Error processing {file}: {e}")

    end = time.time()
    print(f"\nProcessing completed in {end - start:.2f} seconds.")
    master_df = pd.DataFrame(master_records)

    # Reorder columns to guarantee Engine is Column 2
    preferred_first = ["Flight ID", "Engine"]
    remaining = [c for c in master_df.columns if c not in preferred_first]
    master_df = master_df[preferred_first + remaining]

    master_df.to_csv(output_path, index=False)
    print(f"\nMasterFlightList saved to: {output_path}")


# === Example Usage ===
if __name__ == "__main__":
    phases_folder = r"C:\Users\aqeel\OneDrive - The University of Manchester\UOM-RG-FSE-DUST - Aqeel_Shared_Files\ENPROT_PowerBI\Flight Data\Mass Flow Rate"
    masterflightlist = r"C:\Users\aqeel\OneDrive - The University of Manchester\UOM-RG-FSE-DUST - Aqeel_Shared_Files\ENPROT_PowerBI\Flight Data\Flight_Lists"  # or a folder containing multiple CSVs
    output_csv = r"C:\Users\aqeel\OneDrive - The University of Manchester\Desktop\MasterFlightList_with_engine.csv"
    generate_master_flight_list(phases_folder, masterflightlist, output_csv)
