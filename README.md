# ODB2ANY: ABAQUS Output Database Extraction & Visualization Tool

ODB2ANY is a professional toolkit for extracting, converting, and visualizing ABAQUS Output Database (ODB) results. It provides a streamlined workflow for post-processing ABAQUS analysis results through JSON conversion and customizable plotting capabilities.

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Detailed Usage Guide](#detailed-usage-guide)
- [Configuration Files](#configuration-files)
- [Output Structure](#output-structure)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Features

### ODB2JSON (Data Extraction)
- Extract complete field outputs (stress, strain, displacement, etc.)
- Export history outputs with time-series data
- Generate comprehensive header information
- Support for multiple instances and sets
- Automatic metadata inclusion
- Flexible output organization

### ODB2PLOT (Visualization)
- Customizable plot styles and configurations
- Support for various output types
- Range visualization for field outputs
- Time-series plotting for history outputs
- CSV export for further customization
- Built-in data smoothing and processing

## Requirements

### For ODB2JSON (Extraction)
- ABAQUS/CAE with Python 2.7 environment
- NumPy (included in Abaqus Python)
- odbAccess module (provided by Abaqus)

### For ODB2PLOT (Visualization)
- Python 3.x
- NumPy
- Matplotlib

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/AliSaeeidiRad/ODB2ANY.git
   cd ODB2ANY
   ```

2. No additional installation is needed for ODB2JSON as it uses Abaqus's built-in Python.

3. For ODB2PLOT, install required Python packages:
   ```bash
   pip install numpy matplotlib
   ```

## Quick Start

1. Submit your ABAQUS job:
   ```bash
   cd example
   abaqus job=Job-1 cpus=8
   cd ..
   ```

2. Convert ODB results to JSON:
   ```bash
   abaqus python odb2json.py --odb-file example/Job-1.odb --write-history --write-field --instances PART-1-1 --step *
   ```

3. Generate header information:
   ```bash
   python odb2plot.py --field example/ODB2JSON/Job-1_instances_PART-1-1_field_output.json --history example/ODB2JSON/Job-1_history_output.json --header
   ```

4. Plot von Mises stress:
   ```bash
   python odb2plot.py --field example/ODB2JSON/Job-1_instances_PART-1-1_field_output.json --plot-config plot_config_s_example.json --field-option S_ --data-key mises --step *
   ```

![Von Mises Stress Plot Example](example/plot_example_for_vom_mises.tiff)

5. Plot displacements:
   ```bash
   python odb2plot.py --field example/ODB2JSON/Job-1_instances_PART-1-1_field_output.json --plot-config plot_config_u_example.json --field-option U_ --step *
   ```

![U Plot Example](example/plot_example_for_translation.tiff)

## Detailed Usage Guide

### ODB2JSON Commands

1. Extract Header Information:
   ```bash
   abaqus python odb2json.py --odb-file <path-to-odb> --header
   ```

2. Extract History Output:
   ```bash
   abaqus python odb2json.py --odb-file <path-to-odb> --write-history --step *
   ```

3. Extract Field Output:
   ```bash
   abaqus python odb2json.py --odb-file <path-to-odb> --write-field --instances <INSTANCE> --step *
   ```

4. Combined Extraction:
   ```bash
   abaqus python odb2json.py --odb-file <path-to-odb> --write-history --write-field --instances <INSTANCE> --step *
   ```

### Data Selection Options
ODB2JSON supports three types of data extraction from your ABAQUS ODB file:
1. **Element Sets**: Extract data from specific element sets defined in your model
2. **Node Sets**: Extract data from specific node sets defined in your model
3. **Instances**: Extract data from entire part instances in your model

Example commands:
```bash
# Extract from element set
abaqus python odb2json.py --odb-file model.odb --write-field --element-sets SET-1 --step *

# Extract from node set
abaqus python odb2json.py --odb-file model.odb --write-field --node-sets NSET-1 --step *

# Extract from instance
abaqus python odb2json.py --odb-file model.odb --write-field --instances PART-1-1 --step *
```

### ODB2PLOT Commands

1. Extract Headers:
   ```bash
   python odb2plot.py --field <field-json> --history <history-json> --header
   ```

2. Plot Field Output:
   ```bash
   python odb2plot.py --field <field-json> --plot-config <config-json> --field-option <option> --data-key <key> --step *
   ```
   - For specific part selection: `--field-option S_,PART-1`
   - Available data keys: `mises`, `data`, etc.

3. Plot History Output:
   ```bash
   python odb2plot.py --history <history-json> --plot-config <config-json> --history-option <option>
   ```

## Configuration Files

### Plot Configuration
- `plot_config_s_example.json`: Example config for stress plots
- `plot_config_u_example.json`: Example config for displacement plots

Configuration options include:
- Figure size and layout
- Plot colors and styles
- Axis labels and limits
- Grid properties
- Legend settings

## Output Structure

```
example/
└── ODB2JSON/
    ├── Job-1_history_output.json
    ├── Job-1_instances_PART-1-1_field_output.json
    ├── Job-1_instances_PART-1-1_field_output_summary.json
    └── ODB2PLOT/
        ├── HEADERS/
        │   ├── Job-1_history_output.json
        │   └── Job-1_instances_PART-1-1_field_output.json
        └── PLOTS/
            ├── FIELD/
            │   ├── Job-1_instances_PART-1-1_field_output_S_PART-1-1.csv
            │   ├── Job-1_instances_PART-1-1_field_output_S_PART-1-1.tiff
            │   ├── Job-1_instances_PART-1-1_field_output_U_PART-1-1.csv
            │   └── Job-1_instances_PART-1-1_field_output_U_PART-1-1.tiff
            └── HISTORY/
```

## Troubleshooting

1. **ODB2JSON Issues:**
   - Ensure running with `abaqus python`
   - Verify ODB file exists and is not locked
   - Check available memory for large ODB files
   - Verify instance and set names exist in the model

2. **ODB2PLOT Issues:**
   - Ensure Python 3.x is used
   - Check NumPy and Matplotlib installation
   - Verify JSON file paths and structure
   - Check plot configuration format

## Citation

If you use ODB2ANY in your research, please cite it using:

```bibtex
@software{saeedirad2025odb2any,
  author       = {Saeedi Rad, Ali},
  title        = {{ODB2ANY: ABAQUS Output Database Extraction & Visualization Tool}},
  month        = sep,
  year         = 2025,
  publisher    = {GitHub},
  version      = {1.0.0},
  url          = {https://github.com/AliSaeeidiRad/ODB2ANY},
  description  = {A toolkit for extracting, converting, and visualizing ABAQUS Output Database (ODB) results}
}
```

## License

MIT License - Copyright (c) 2025 Ali Saeedi Rad

## Author
Ali Saeedi Rad  
Version: 1.0.0  
Date: September 2025  
Repository: https://github.com/AliSaeeidiRad/ODB2ANY