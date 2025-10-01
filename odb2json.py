"""
ODB2JSON: A powerful utility for extracting ABAQUS Output Database (ODB) data to JSON format.

This module provides comprehensive functionality to extract and convert various types of data
from ABAQUS ODB files into structured JSON format for further analysis and visualization.
It is designed to work seamlessly with the Abaqus Python interpreter and provides robust
error handling and data validation.

Features:
    - Header Information Export:
        * Instance names and properties
        * Node and element set definitions
        * Analysis steps and frames structure
        * Available field and history outputs

    - Field Output Extraction:
        * Support for nodal and element data
        * Multiple instance/set processing
        * Comprehensive metadata inclusion
        * Optional data filtering and summarization

    - History Output Processing:
        * Time-series data extraction
        * Region-based output organization
        * Consistent data formatting
        * Automatic missing data handling

    - General Features:
        * Command-line interface
        * JSON schema versioning
        * Flexible output organization
        * Progress logging and error reporting

Requirements:
    - Abaqus/CAE environment with Python interpreter
    - Python packages:
        * NumPy (included in Abaqus Python)
        * odbAccess module (provided by Abaqus)

Usage Examples:
    Extract header information:
        abaqus python odb2json.py --odb-file model.odb --header

    Extract field output for specific instance:
        abaqus python odb2json.py --odb-file model.odb --write-field -i Part-1-1

    Extract history output for all steps:
        abaqus python odb2json.py --odb-file model.odb --write-history --step "*"

    Combined extraction with custom output directory:
        abaqus python odb2json.py --odb-file model.odb --write-field --write-history
                                 -i Part-1-1 -o ./results --step "Step-1:0,1,2"

Author: Ali Saeedi Rad
License: MIT License
Version: 1.0.0
Date: September 2025
Repository: https://github.com/AliSaeeidiRad/ODB2ANY
"""

import os
import sys
import json
import logging
import argparse

try:
    import numpy as np
except ImportError:
    print("NumPy is not installed. This script requires NumPy to function properly.")
    sys.exit(1)

try:
    from odbAccess import openOdb  # type: ignore
except ImportError:
    print(
        "odbAccess module not found. This script must be run using "
        "the Abaqus Python interpreter."
    )
    sys.exit(1)

# Version and schema information
__version__ = "1.0.0"
__schema_version__ = "1.0.0"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class ODB:
    """ABAQUS Output Database (ODB) wrapper class for simplified data extraction.

    This class provides a high-level interface to interact with ABAQUS ODB files,
    abstracting away the complexity of direct odbAccess API usage. It offers methods
    to access and extract various types of data including:
        - Analysis steps and frames
        - Model instances and their components
        - Element and node sets
        - Field outputs (stress, strain, displacement, etc.)
        - History outputs (time-series data)

    The class implements proper resource management and error handling to ensure
    safe ODB file operations.

    Attributes:
        filename (Path): Path to the ODB file.
        _odb (Any): Internal reference to the opened odbAccess.Odb object.
            Type is Any to avoid dependency on odbAccess in type checking.

    Example:
        >>> from odb2json import *
        >>> odb = ODB("analysis.odb")
        >>> steps = odb.get_steps_keys
        >>> frames = odb.get_frames(steps[0])
        >>> field_data = odb.get_field_output("Step-1", 0, "S")
    """

    def __init__(self, odb, read_only=True):
        """Initialize and open an ODB file.

        Args:
            odb (str): Path to the ODB file.
            read_only (bool): If True, opens file in read-only mode. Defaults to True.

        Raises:
            IOError: If the ODB file doesn't exist.
            RuntimeError: If odbAccess fails to open the file.
        """
        self.filename = odb
        if not os.path.exists(self.filename):
            raise IOError("ODB file not found: %s" % self.filename)

        try:
            self._odb = openOdb(self.filename, read_only)
        except Exception as e:
            raise RuntimeError("Failed to open ODB file: %s" % str(e))

    def get_frames(self, step_name):
        """Retrieve all frames for a given analysis step.

        This method provides access to all available frames within a specific
        analysis step. Each frame represents a point in time where results
        were written during the analysis.

        Args:
            step_name (str): Name of the analysis step.

        Returns:
            list: List of Frame objects from the specified step.

        Raises:
            KeyError: If the step name is not found in the ODB.
        """
        try:
            return self._odb.steps[step_name].frames
        except KeyError:
            raise KeyError("Step '%s' not found in ODB file" % step_name)

    def get_frame(self, step_name, frame_num):
        """Retrieve a specific frame from an analysis step.

        This method provides direct access to a single frame within a step,
        allowing for targeted data extraction from specific points in the analysis.

        Args:
            step_name (str): Name of the analysis step.
            frame_num (int): Index of the frame to retrieve.

        Returns:
            Frame: The requested Frame object.

        Raises:
            KeyError: If the step name is not found.
            IndexError: If the frame index is out of range.
        """
        frames = self.get_frames(step_name)
        try:
            return frames[frame_num]
        except IndexError:
            raise IndexError(
                "Frame %d not found in step '%s'. Available frames: 0-%d"
                % (frame_num, step_name, len(frames) - 1)
            )

    def get_instance(self, instance_name):
        """Retrieve a specific instance from the root assembly.

        Model instances represent different parts or components in the analysis.
        This method provides access to instance data including geometry and mesh.

        Args:
            instance_name (str): Name of the instance to retrieve.

        Returns:
            Instance: The requested Instance object.

        Raises:
            KeyError: If the instance name is not found in the model.
        """
        try:
            return self._odb.rootAssembly.instances[instance_name]
        except KeyError:
            available = list(self._odb.rootAssembly.instances.keys())
            raise KeyError(
                "Instance '%s' not found. Available instances: %s"
                % (instance_name, available)
            )

    def get_elementsets(self, elementset_name):
        """Retrieve an element set by name from the root assembly.

        Element sets are named collections of elements used for result extraction
        and post-processing. This method provides access to these sets.

        Args:
            elementset_name (str): Name of the element set to retrieve.

        Returns:
            ElementSet: The requested ElementSet object.

        Raises:
            KeyError: If the element set name is not found.
        """
        try:
            return self._odb.rootAssembly.elementSets[elementset_name]
        except KeyError:
            available = list(self._odb.rootAssembly.elementSets.keys())
            raise KeyError(
                "Element set '{}' not found. ".format(elementset_name),
                "Available element sets: {available}".format(available),
            )

    def get_nodesets(self, nodeset_name):
        """Retrieve a node set by name from the root assembly.

        Node sets are named collections of nodes used for result extraction
        and post-processing. This method provides access to these sets.

        Args:
            nodeset_name (str): Name of the node set to retrieve.

        Returns:
            NodeSet: The requested NodeSet object.

        Raises:
            KeyError: If the node set name is not found.
        """
        try:
            return self._odb.rootAssembly.nodeSets[nodeset_name]
        except KeyError:
            available = list(self._odb.rootAssembly.nodeSets.keys())
            raise KeyError(
                "Node set '%s' not found. Available node sets: %s"
                % (nodeset_name, available)
            )

    def get_nodes(self, instance_name):
        """Retrieve all nodes associated with a given instance.

        This method provides access to the complete set of nodes that make up
        the mesh of a specific model instance.

        Args:
            instance_name (str): Name of the instance whose nodes to retrieve.

        Returns:
            sequence: A sequence of Node objects.

        Raises:
            KeyError: If the instance name is not found.
        """
        instance = self.get_instance(instance_name)
        return instance.nodes

    def get_elements(self, instance_name):
        """
        Retrieves the elements associated with a given instance.

        Args:
            instance_name (str): Name of the instance.

        Returns:
            sequence: A sequence of Element objects.
        """
        return self.get_instance(instance_name).elements

    def get_field_output_keys(self, step_name, frame_idx):
        """
        Returns the available field output keys for a specific frame.

        Args:
            step_name (str): Name of the step.
            frame_idx (int): Index of the frame.

        Returns:
            list: A list of field output names.
        """
        return self._odb.steps[step_name].frames[frame_idx].fieldOutputs.keys()

    def get_field_output(self, step_name, frame_idx, fld_name):
        """
        Retrieves a specific field output from a given frame.

        Args:
            step_name (str): Name of the step.
            frame_idx (int): Index of the frame.
            fld_name (str): Name of the field output.

        Returns:
            odbAccess.FieldOutput: The requested field output.
        """
        return self._odb.steps[step_name].frames[frame_idx].fieldOutputs[fld_name]

    def get_field_outputs(self, step_name, frame_idx):
        """
        Retrieves all field outputs for a given frame.

        Args:
            step_name (str): Name of the step.
            frame_idx (int): Index of the frame.

        Returns:
            list: A list of tuples (field name, FieldOutput).
        """
        return self._odb.steps[step_name].frames[frame_idx].fieldOutputs.items()

    def get_history_regions(self, step_name):
        """
        Retrieves the history regions for a given step.

        Args:
            step_name (str): Name of the step.

        Returns:
            dict: A dictionary of HistoryRegion objects keyed by region name.
        """
        return self._odb.steps[step_name].historyRegions

    @property
    def odb(self):
        """
        Provides direct access to the underlying ODB object.

        Returns:
            odbAccess.Odb: The underlying ODB instance.
        """
        return self._odb

    @property
    def get_steps_keys(self):
        """
        Lists the names of all steps in the ODB.

        Returns:
            list: A list of step names.
        """
        return self._odb.steps.keys()

    @property
    def get_instances_keys(self):
        """
        Lists the names of all instances in the root assembly.

        Returns:
            list: A list of instance names.
        """
        return self._odb.rootAssembly.instances.keys()

    @property
    def get_elementsets_keys(self):
        """
        Lists the names of all element sets in the root assembly.

        Returns:
            list: A list of element set names.
        """
        return self._odb.rootAssembly.elementSets.keys()

    @property
    def get_nodesets_keys(self):
        """
        Lists the names of all node sets in the root assembly.

        Returns:
            list: A list of node set names.
        """
        return self._odb.rootAssembly.nodeSets.keys()


def extract_headers_information(odb, output_dir):
    """
    Extract header information from an ODB (Output Database) file and export it as a structured JSON file.

    This function collects high-level metadata from the given ODB object, including:
      - Instance names
      - Element set names
      - Node set names
      - Step names along with their associated frame identifiers

    The extracted information is organized into a dictionary and saved to a JSON file
    located in the following directory:
        <ODB_DIR>/ODB2JSON/HEADERS/

    The output filename follows this pattern:
        <original_odb_filename>.json

    Parameters:
    -----------
    odb : ODB
        An instance of the `ODB` class, which serves as a wrapper to facilitate structured
        interaction with Abaqus ODB files.

    output_dir: str
        Output directory of results

    Output:
    -------
    A JSON file containing the following structure:
    {
        "instances": [...],
        "elementSets": [...],
        "nodeSets": [...],
        "steps": [
            ["Step-1", ["Step-1-frame-000", "Step-1-frame-001", ...]],
            ...
        ]
    }

    Notes:
    ------
    - If the output directory already exists, it will not raise an error, and the existing file will be overwritten.
    - A message will be printed with the absolute path to the exported JSON file.
    """
    dict_json = {
        "instances": [],
        "elementSets": [],
        "nodeSets": [],
        "steps": [],
    }

    dict_json["instances"].extend(odb.get_instances_keys)
    dict_json["elementSets"].extend(odb.get_elementsets_keys)
    dict_json["nodeSets"].extend(odb.get_nodesets_keys)

    for step_name in odb.get_steps_keys:
        step_frame_map = [step_name]
        frames = []
        frame_count = len(odb.get_frames(step_name))
        for i in range(frame_count):
            frame_name = "{}-frame-{}".format(
                step_name, str(i).zfill(len(str(frame_count)))
            )
            frames.append(frame_name)
        step_frame_map.append(frames)
        dict_json["steps"].append(step_frame_map)

    output_dir = os.path.join(
        os.path.dirname(odb.filename) if output_dir == "" else output_dir,
        "ODB2JSON",
        "HEADERS",
    )

    try:
        os.makedirs(output_dir)
    except:
        print("Directory Already Exists.")

    fname = os.path.join(
        output_dir,
        os.path.basename(odb.filename).replace(".odb", ".json"),
    )
    with open(fname, "w") as fp:
        json.dump(dict_json, fp, indent=4)

    print('File Exported: "{}"'.format(os.path.abspath(fname)))


def write_history_output(odb, steps, output_dir):
    """
    Extracts history output data from an ODB file and writes it to a JSON file.

    This function processes history output data across specified steps and
    organizes them into a tabular format where each column represents a specific
    history variable from a region (e.g., node, element, global output), and
    each row represents a data point in time.

    Parameters:
    -----------
    odb : ODB
        An instance of the `ODB` class, which serves as a wrapper to facilitate structured
        interaction with Abaqus ODB files.

    steps : dict
        A dictionary where keys are step names (str) that should be processed.
        The values are not used, but keys determine which steps to extract.

    output_dir: str
        Output directory of results

    Output:
    -------
    A JSON file saved in the directory:
        <ODB_DIR>/ODB2JSON/

    The output filename will be:
        <original_odb_filename>_history_output.json

    The JSON will have the format:
    {
        "Step-1_Region-1_Output-1": [val1, val2, ...],
        "Step-1_Region-2_Output-2": [val1, val2, ...],
        ...
    }

    Notes:
    ------
    - The function automatically determines the number of data points.
    - If the output directory already exists, it is reused.
    - Data arrays are aligned to the longest available output; missing entries are filled with 0.
    - A message is printed with the full path to the exported file.
    """
    num_of_data_array = 0
    num_of_data_point = 0

    if steps == "*":
        steps = {}
        for _step in odb.get_steps_keys:
            steps[_step] = list(range(len(odb.get_frames(_step))))

    for step_name in steps.keys():
        history_regions = odb.get_history_regions(step_name)
        if history_regions:  # Check if history regions exist
            for history_region_name, history_region_obj in history_regions.items():
                for (
                    history_output_name,
                    history_output_obj,
                ) in history_region_obj.historyOutputs.items():
                    num_of_data_array += 1
                    num_of_data_point = max(
                        num_of_data_point, len(history_output_obj.data)
                    )

    data = np.zeros((num_of_data_point, num_of_data_array))
    header = []
    col_index = 0  # Keep track of the column index

    for step_name in steps.keys():
        history_regions = odb.get_history_regions(step_name)
        if history_regions:  # Check if history regions exist
            for history_region_name, history_region_obj in history_regions.items():
                for (
                    history_output_name,
                    history_output_obj,
                ) in history_region_obj.historyOutputs.items():
                    name = "{}_{}_{}".format(
                        step_name, history_region_name, history_output_name
                    )
                    header.append(name)
                    for row_index, d in enumerate(history_output_obj.data):
                        data[row_index, col_index] = d[1]
                    col_index += 1  # Increment column index

    output_dir = os.path.join(
        os.path.dirname(odb.filename) if output_dir == "" else output_dir,
        "ODB2JSON",
    )

    try:
        os.makedirs(output_dir)
    except:
        print("Directory Already Exists.")

    fname = os.path.join(
        output_dir,
        os.path.basename(odb.filename).replace(".odb", "_history_output.json"),
    )
    with open(fname, "w") as f:
        json.dump(
            {key: value.tolist() for key, value in zip(header, data.T)}, f, indent=4
        )  # Transpose data

    print('File Exported: "{}"'.format(os.path.abspath(fname)))


def write_field_output(odb, instances, node_sets, element_sets, steps, output_dir):
    """
    Extracts field output data from the specified steps and frames in an Abaqus ODB file,
    for given instances, node sets, or element sets, and writes detailed and summary
    information into JSON files.

    This function processes each frame in each step specified in the `steps` dictionary,
    and extracts field output data (e.g., stress, strain) related to selected regions.
    Two JSON files are created:
        - One containing complete data arrays for each variable
        - One summary file containing the available data keys per region and field

    Parameters:
    -----------
    odb : ODB
        An instance of the `ODB` class, which serves as a wrapper to facilitate structured
        interaction with Abaqus ODB files.

    instances : list[str] or None
        List of instance names to extract field output from. If None, treated as empty.

    node_sets : list[str] or None
        List of node set names to extract field output from. If None, treated as empty.

    element_sets : list[str] or None
        List of element set names to extract field output from. If None, treated as empty.

    steps : dict[str, list[int]]
        Dictionary mapping step names to a list of frame indices to extract.

    output_dir: str
        Output directory of results

    Output:
    -------
    Two JSON files are written to:
        <ODB_DIR>/ODB2JSON/

    1. Field data:
        <odb_basename>_<object_names>_field_output.json
        Contains raw data per step, frame, field name, and region.

    2. Summary data:
        <odb_basename>_<object_names>_field_output_summary.json
        Lists available keys (e.g., 'data', 'position', 'mises') per variable/region per frame.

    Notes:
    ------
    - The function supports extracting complex block data including optional 'mises' and section points.
    - Regions not found in the output field are skipped with an error message.
    - The output directory is created if it does not exist.
    - Two print statements indicate the paths to exported files.
    """
    instances = instances if instances is not None else []
    node_sets = node_sets if node_sets is not None else []
    element_sets = element_sets if element_sets is not None else []
    all_objects = instances + node_sets + element_sets

    if steps == "*":
        steps = {}
        for _step in odb.get_steps_keys:
            steps[_step] = list(range(len(odb.get_frames(_step))))

    data = {}
    summary_data = {}
    for step_name, frame_list in steps.items():
        data[step_name] = {}
        summary_data[step_name] = {}
        for frame_idx in frame_list:
            frame_data = {}
            frame = odb.get_frame(step_name, frame_idx)
            for fld_name, fld_output in frame.fieldOutputs.items():
                for object_name in all_objects:
                    instance_data = []
                    try:
                        if object_name in instances:
                            subset = fld_output.getSubset(
                                region=odb.get_instance(object_name)
                            )
                        elif object_name in element_sets:
                            subset = fld_output.getSubset(
                                region=odb.get_elementsets(object_name)
                            )
                        elif object_name in node_sets:
                            subset = fld_output.getSubset(
                                region=odb.get_nodesets(object_name)
                            )
                        else:
                            raise ValueError(
                                "Given isntance/element sets could not be found. available instances are {} and available element sets are {}.".format(
                                    odb.get_instances_keys,
                                    odb.get_elementsets_keys,
                                )
                            )
                    except Exception as e:
                        print(
                            "Field {} doesn't contain data for instance {}. Skipping. Error: {}".format(
                                fld_name, object_name, e
                            )
                        )
                        continue

                    for block in subset.bulkDataBlocks:
                        block_data = {
                            "data": block.data.tolist(),
                            "position": str(block.position),
                        }
                        if hasattr(block, "mises"):
                            block_data["mises"] = {
                                "mises": (
                                    block.mises.tolist()
                                    if block.mises is not None
                                    else None
                                ),
                            }
                        if block.sectionPoint:
                            block_data["sectionPoint"] = {
                                "description": block.sectionPoint.description,
                            }
                        instance_data.append(block_data)
                    if instance_data:
                        frame_data[fld_name + "_" + object_name] = instance_data
            data[step_name]["frame_{}".format(frame_idx)] = frame_data
            summary_data[step_name]["frame_{}".format(frame_idx)] = []
            for key, value in frame_data.items():
                summary_data[step_name]["frame_{}".format(frame_idx)].append({key: []})
                for _value in value:
                    summary_data[step_name]["frame_{}".format(frame_idx)][-1][key] = (
                        list(_value.keys())
                    )

    output_dir = os.path.join(
        os.path.dirname(odb.filename) if output_dir == "" else output_dir,
        "ODB2JSON",
    )

    try:
        os.makedirs(output_dir)
    except:
        print("Directory Already Exists.")

    _prefix = ""
    if instances:
        _prefix += "instances_{}".format("_".join(instances))
    if element_sets:
        _prefix += "elementsets_{}".format("_".join(element_sets))
    if node_sets:
        _prefix += "nodesets_{}".format("_".join(node_sets))

    fname_data = os.path.join(
        output_dir,
        os.path.basename(odb.filename).replace(
            ".odb", "_{}_field_output.json".format(_prefix)
        ),
    )

    with open(fname_data, "w") as f:
        json.dump(data, f, indent=4)

    print('File Exported: "{}"'.format(os.path.abspath(fname_data)))

    fname_summary_data = os.path.join(
        output_dir,
        os.path.basename(odb.filename).replace(
            ".odb", "_{}_field_output_summary.json".format(_prefix)
        ),
    )

    with open(fname_summary_data, "w") as f:
        json.dump(summary_data, f, indent=4)

    print('File Exported: "{}"'.format(os.path.abspath(fname_summary_data)))


def parse_arguments():
    """Parse and validate command line arguments for ODB data extraction.

    This function sets up the command-line interface for the script, organizing
    arguments into logical groups and providing detailed help information.

    Returns:
        argparse.Namespace: Parsed command line arguments.

    Example:
        To extract header information:
        abaqus python odb2json.py --odb-file model.odb --header

        To extract field output:
        abaqus python odb2json.py --odb-file model.odb --write-field -i Part-1-1 --step "Step-1:0,1,2"

        To extract history output:
        abaqus python odb2json.py --odb-file model.odb --write-history --step "*"
    """
    parser = argparse.ArgumentParser(
        prog="odb2json",
        description="Professional ABAQUS ODB data extraction utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input options
    input_group = parser.add_argument_group("Input Options")
    input_group.add_argument(
        "-of",
        "--odb-file",
        required=True,
        metavar="FILE",
        help="Path to the input ABAQUS ODB file",
    )
    input_group.add_argument(
        "--header",
        action="store_true",
        help="Extract header information only (instances, steps, sets)",
    )

    # Data extraction options
    extraction_group = parser.add_argument_group("Data Extraction Options")
    extraction_group.add_argument(
        "-wh",
        "--write-history",
        action="store_true",
        help="Extract history output data",
    )
    extraction_group.add_argument(
        "-wf", "--write-field", action="store_true", help="Extract field output data"
    )

    # Region selection
    region_group = parser.add_argument_group("Region Selection")
    region_group.add_argument(
        "-i",
        "--instances",
        nargs="+",
        metavar="INSTANCE",
        help="Instance names to process (e.g., Part-1-1 Part-2-1)",
    )
    region_group.add_argument(
        "-es",
        "--element-sets",
        nargs="+",
        metavar="SET",
        help="Element set names to process (e.g., SET-1 SET-2)",
    )
    region_group.add_argument(
        "-ns",
        "--node-sets",
        nargs="+",
        metavar="SET",
        help="Node set names to process (e.g., NSET-1 NSET-2)",
    )

    # Step and frame selection
    step_group = parser.add_argument_group("Step and Frame Selection")
    step_group.add_argument(
        "-st",
        "--step",
        nargs="+",
        metavar="STEP:FRAMES",
        help='Steps and frames to process (e.g., "Step-1:0,1,2" or "*" for all)',
    )

    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "-o",
        "--output-dir",
        default="",
        metavar="DIR",
        help="Output directory for JSON files (default: ODB file location)",
    )
    output_group.add_argument(
        "-sf",
        "--suffix",
        default="",
        metavar="SUFFIX",
        help="Suffix to append to output filenames",
    )

    args = parser.parse_args()

    # Validate argument combinations
    if not args.header and not args.write_history and not args.write_field:
        parser.error(
            "At least one action (--header, --write-history, --write-field) must be specified"
        )

    if args.write_field and not any(
        [args.instances, args.element_sets, args.node_sets]
    ):
        parser.error(
            "Field output extraction requires at least one region (--instances, --element-sets, or --node-sets)"
        )

    if (args.write_field or args.write_history) and not args.step:
        parser.error(
            "Step selection (--step) is required for field or history output extraction"
        )

    return args


def main():
    """Main entry point for the ODB to JSON conversion utility.

    This function orchestrates the entire data extraction process based on
    command-line arguments. It handles:
        - Argument parsing and validation
        - ODB file opening
        - Data extraction coordination
        - Error handling and logging
        - Progress reporting

    The function implements a structured workflow that:
        1. Validates input arguments
        2. Opens the ODB file safely
        3. Processes requested data types
        4. Handles any errors gracefully with informative messages

    Raises:
        IOError: If the ODB file doesn't exist
        ValueError: If required arguments are missing or invalid
        RuntimeError: If there are issues with ODB access or processing
    """
    try:
        logger.info("Starting ODB2JSON extraction process")
        args = parse_arguments()

        logger.info("Processing ODB file: %s" % args.odb_file)
        odb = ODB(args.odb_file)

        if args.header:
            logger.info("Extracting header information")
            extract_headers_information(odb, args.output_dir)
            logger.info("Header extraction completed")
            return

        # Process step specifications
        logger.info("Processing step specifications")
        if args.step != ["*"]:
            steps = {}
            for item in args.step:
                try:
                    step_name, frame_str = item.split(":")
                    frame_indices = [int(f) for f in frame_str.split(",")]
                    steps[step_name] = frame_indices
                except ValueError:
                    raise ValueError(
                        "Invalid step format: %s. Expected format: 'StepName:0,1,2'"
                        % item
                    )
        else:
            steps = "*"

        # Extract history output if requested
        if args.write_history:
            logger.info("Starting history output extraction")
            write_history_output(odb, steps, args.output_dir)
            logger.info("History output extraction completed")

        # Extract field output if requested
        if args.write_field:
            logger.info("Starting field output extraction")
            write_field_output(
                odb,
                args.instances,
                args.node_sets,
                args.element_sets,
                steps,
                args.output_dir,
            )
            logger.info("Field output extraction completed")

        logger.info("ODB2JSON extraction process completed successfully")

    except IOError as e:
        logger.error("File not found: %s" % str(e))
        raise

    except ValueError as e:
        logger.error("Invalid input: %s" % str(e))
        raise

    except Exception as e:
        logger.error("An unexpected error occurred: %s" % str(e))
        raise

    finally:
        logger.info("ODB2JSON process finished")


if __name__ == "__main__":
    try:
        import odbAccess  # type: ignore
    except:
        raise ModuleNotFoundError(
            "Script has no access to 'odbAccess'. Make sure you have Abaqus/CAE installed and run this script using 'abaqus python odb2json.py ...'"
        )
    main()
