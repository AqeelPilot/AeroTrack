import os
import json
import tkinter as tk
from tkinter import filedialog


from Flight_Program_Lookup_Modules import (
    FlightProcessor,
    MasterFlightLookup,
    AircraftPropertiesLookup,
    MiscUtilities,
    Plotting,
    log_error,
)


...


# ========================= WELCOME MESSAGE ========================= #
print("\033[95m" + "=" * 65)
print("\033[94m" + " " * 23 + "FLIGHT PROGRAM")
print("\033[96m" + " " * 10 + "Developed by Muhammad Aqeel Abdulla")
print("\033[95m" + "=" * 65 + "\033[0m")
print("\n")
# ================================================================ #

if __name__ == "__main__":
    print("Flight Processor Started")

    # === GUI to Select Config File ===
    root = tk.Tk()
    root.withdraw()  # Hide tkinter root window

    print("Please select your 'project_config.json' file...")
    config_path = filedialog.askopenfilename(
        title="Select project_config.json", filetypes=[("JSON files", "*.json")]
    )

    if not config_path:
        print("No config file selected. Exiting.")
        exit(1)

    print(f"Loaded config from: {config_path}")

    # === Load Configuration ===
    with open(config_path, "r") as f:
        config = json.load(f)

    input_path = config.get("input_folder")
    output_path = config.get("output_folder")
    master_list_path = config.get("master_flight_list_folder")
    project_name = config.get("project_name")
    print("PROJECT NAME:", project_name)
    print(f"Input folder: {input_path}")
    print(f"Output folder: {output_path}")
    print(f"Master flight list: {master_list_path}\n")

    # === Initialize Modules ===
    processor = FlightProcessor(input_path, output_path, project_name, config_path)
    lookup = MasterFlightLookup(master_list_path)
    aircraft_lookup = AircraftPropertiesLookup(lookup)

    # === Process Files ===
    csv_files = [f for f in os.listdir(input_path) if f.endswith(".csv")]
    print(f"Found {len(csv_files)} flight files.\n")

    for i, file_name in enumerate(csv_files, start=1):
        try:
            processor.process_file(
                file_name, i, len(csv_files), lookup, aircraft_lookup
            )
        except Exception as e:
            log_error(file_name, f"Processing error: {str(e)}")

    print("\nAll files processed successfully.")
    # === Initialize Plotting Class and Launch Plot ===
    # plotter = Plotting(output_path)
    # plotter.plot_drag_vs_time()
