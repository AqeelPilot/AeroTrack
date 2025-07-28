Installation
============

This page describes how to install and set up **AeroTrack**, a post-flight analysis toolkit for processing aircraft trajectory data.

System Requirements
-------------------

- Python 3.10 or later
- Operating System: Windows, macOS, or Linux
- Internet access (required for first-time OpenAP use)
- Optional: `tkinter` for GUI file dialog (usually pre-installed)

Python Dependencies
-------------------

AeroTrack depends on the following Python packages:

- `openap`
- `pandas`
- `numpy`
- `matplotlib`
- `scipy`
- `tkinter` (for GUI input)
- `datetime`, `os`, `json`, `logging`, `time` (standard library)

It is recommended to install all dependencies via a `requirements.txt` file.

Installation Steps
------------------

1. **Clone the Repository**

If AeroTrack is hosted in a Git repository:

.. code-block:: bash

    git clone https://github.com/YourUsername/AeroTrack.git
    cd AeroTrack

2. **Create a Virtual Environment**

.. code-block:: bash

    python -m venv venv
    source venv/bin/activate   # On Windows: venv\Scripts\activate

3. **Install Dependencies**

.. code-block:: bash

    pip install -r requirements.txt

If you do not have a `requirements.txt`, you can install dependencies manually:

.. code-block:: bash

    pip install openap pandas numpy matplotlib scipy

4. **Install Tkinter (If Missing)**

Tkinter is required for the GUI file selection prompt:

- **Windows/macOS**: usually included with Python.
- **Linux (Debian/Ubuntu)**:

.. code-block:: bash

    sudo apt-get install python3-tk

Running AeroTrack
-----------------

Once installed, run the main entry point:

.. code-block:: bash

    python AeroTrack_MAIN.py

You will be prompted to select a `project_config.json` file using a file dialog.

Project Configuration File
--------------------------

The configuration file must contain the following keys:

.. code-block:: json

    {
      "input_folder": "path/to/csv_inputs",
      "output_folder": "path/to/save_outputs",
      "master_flight_list_folder": "path/to/masterlist",
      "project_name": "MyProject"
    }

Folder Structure
----------------

The AeroTrack runtime expects the following folder structure:

.. code-block:: text

    project_root/
    ├── input_folder/                  # Contains raw flight .csv files
    ├── output_folder/                 # Destination for processed files
    ├── master_flight_list_folder/     # Contains master .csv with aircraft details
    ├── project_config.json            # Configuration file (selected at runtime)
    ├── AeroTrack_MAIN.py
    ├── AeroTrack_Modules.py

After selecting the config file, AeroTrack will automatically:

- Process all `.csv` flight logs in the input folder
- Annotate files with drag, fuel, and flight phase data
- Save enhanced `.csv` files to the output folder
- Optionally generate visualizations (if enabled)

Next Step
---------

Once installation is complete, proceed to the :doc:`usage` section to learn how to run AeroTrack and interpret outputs.
