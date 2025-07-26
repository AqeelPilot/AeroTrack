import os
import json
import tkinter as tk
from tkinter import filedialog
import time


from AeroTrack_Modules import (
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
print("\033[94m" + " " * 23 + "AeroTrack")
print("\033[96m" + " " * 10 + "Developed by Muhammad Aqeel Abdulla")
print("\033[95m" + "=" * 65 + "\033[0m")
print("\n\033[0m" + "AeroTrack is a post-flight analysis tool designed to extract")
print("aerodynamic and performance insights from aircraft trajectory data.")
print("It integrates real-world flight CSVs with modeled performance data")
print("to estimate key metrics like drag, fuel consumption, and flight phase.")
print("\nThe system automatically:")
print(" - Calculates flight path angle and vertical speed")
print(" - Estimates total drag and weight components using OpenAP")
print(" - Identifies flight phases and cruise segments")
print(" - Infers fuel burn using engine-specific models")
print(
    "\nAeroTrack serves as the analytical backbone of the Dust Flight Dashboard,\nproviding high-fidelity physical interpretation of aircraft behavior."
)
print("\033[95m" + "=" * 65 + "\033[0m\n")

# ================================================================ #

if __name__ == "__main__":
    start = time.time()
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
    csv_files = [f for f in os.listdir(input_path) if f.endswith(".csv")]
    print(f"Found {len(csv_files)} flight files.\n")
    # === Initialize Modules ===
    use_openap = int(input("Use OpenAP dynamic fuel flow model? (1/0): "))
    processor = FlightProcessor(
        input_path, output_path, project_name, config_path, use_openap
    )
    lookup = MasterFlightLookup(master_list_path)
    aircraft_lookup = AircraftPropertiesLookup(lookup)

    # === Process Files ===

    for i, file_name in enumerate(csv_files, start=1):
        try:
            processor.process_file(
                file_name, i, len(csv_files), lookup, aircraft_lookup
            )
        except Exception as e:
            log_error(file_name, f"Processing error: {str(e)}")
    end = time.time()
    print(f"\nProcessing completed in {end - start:.2f} seconds.")
    print("\nAll files processed successfully.")
    # === Initialize Plotting Class and Launch Plot ===
    # plotter = Plotting(output_path)
    # plotter.plot_drag_vs_time()
