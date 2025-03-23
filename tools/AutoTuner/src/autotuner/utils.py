#############################################################################
##
## BSD 3-Clause License
##
## Copyright (c) 2019, The Regents of the University of California
## All rights reserved.
##
## Redistribution and use in source and binary forms, with or without
## modification, are permitted provided that the following conditions are met:
##
## * Redistributions of source code must retain the above copyright notice, this
##   list of conditions and the following disclaimer.
##
## * Redistributions in binary form must reproduce the above copyright notice,
##   this list of conditions and the following disclaimer in the documentation
##   and/or other materials provided with the distribution.
##
## * Neither the name of the copyright holder nor the names of its
##   contributors may be used to endorse or promote products derived from
##   this software without specific prior written permission.
##
## THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
## AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
## IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
## ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
## LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
## CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
## SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
## INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
## CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
## ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
## POSSIBILITY OF SUCH DAMAGE.
##
###############################################################################

import glob
import json
import os
import re
import yaml
import subprocess
import sys
from multiprocessing import cpu_count
from datetime import datetime
from uuid import uuid4 as uuid
from time import time
from copy import deepcopy
from shutil import copytree, ignore_patterns

import numpy as np
import ray
from ray import tune

# Default scheme of a SDC constraints file
SDC_TEMPLATE = """
set clk_name  core_clock
set clk_port_name clk
set clk_period 2000
set clk_io_pct 0.2

set clk_port [get_ports $clk_port_name]

create_clock -name $clk_name -period $clk_period $clk_port

set non_clock_inputs [lsearch -inline -all -not -exact [all_inputs] $clk_port]

set_input_delay  [expr $clk_period * $clk_io_pct] -clock $clk_name $non_clock_inputs
set_output_delay [expr $clk_period * $clk_io_pct] -clock $clk_name [all_outputs]
"""

JSON_FILEPATHS = { # Base path for files
    "_SDC_FILE_PATH": None,
    "_FR_FILE_PATH": None,
    "_PACKAGE_FILE_PATH": None,
    "_TOP_LEVEL_FILE_PATH": None,
    "_SIM_FILE_PATH": None
}
# Path to the FLOW_HOME directory
ORFS_FLOW_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../../flow")
)
DATE = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
ALLOWED_OPENROAD_STAGES = ["floorplan", "place", "cts", "route", "all", "none"]
METRICS_CONFIG = dict() # Configuration for the metrics used to compute the score for each run


def write_sdc(variables, sdc_original, filepath):
    """
    Create a SDC file with parameters for current tuning iteration.
    """

    # Handle case where the reference file does not exist
    if sdc_original == "":
        print("[ERROR TUN-0020] No SDC reference file provided.")
        sys.exit(1)
    new_file = sdc_original
    for key, value in variables.items():
        if key == "CLK_PERIOD":
            if new_file.find("set clk_period") != -1:
                new_file = re.sub(
                    r"set clk_period .*\n(.*)", f"set clk_period {value}\n\\1", new_file
                )
            else:
                new_file = re.sub(
                    r"-period [0-9\.]+ (.*)", f"-period {value} \\1", new_file
                )
                new_file = re.sub(r"-waveform [{}\s0-9\.]+[\s|\n]", "", new_file)
        elif key == "UNCERTAINTY":
            if new_file.find("set uncertainty") != -1:
                new_file = re.sub(
                    r"set uncertainty .*\n(.*)",
                    f"set uncertainty {value}\n\\1",
                    new_file,
                )
            else:
                new_file += f"\nset uncertainty {value}\n"
        elif key == "IO_DELAY":
            if new_file.find("set io_delay") != -1:
                new_file = re.sub(
                    r"set io_delay .*\n(.*)", f"set io_delay {value}\n\\1", new_file
                )
            else:
                new_file += f"\nset io_delay {value}\n"
        else:
            print(
                f"[WARN TUN-0025] {key} variable not supported in context of SDC files"
            )
            continue

    with open(filepath, "w") as file:
        file.write(new_file)


def write_fast_route(variables, platform, fr_original, filepath):
    """
    Create a FastRoute Tcl file with parameters for current tuning iteration.
    """
    # Handle case where the reference file does not exist (asap7 doesn't have reference)
    if fr_original == "" and platform != "asap7":
        print("[ERROR TUN-0021] No FastRoute Tcl reference file provided.")
        sys.exit(1)
    layer_cmd = "set_global_routing_layer_adjustment"
    new_file = fr_original
    # This is part of the defaults when no FASTROUTE_TCL is provided
    if len(new_file) == 0:
        new_file = "set_routing_layers -signal $::env(MIN_ROUTING_LAYER)-$::env(MAX_ROUTING_LAYER)"
    for key, value in variables.items():
        if key.startswith("LAYER_ADJUST"):
            layer = key.lstrip("LAYER_ADJUST")
            # If there is no suffix (i.e., layer name) apply adjust to all
            # layers.
            if layer == "":
                new_file += "\nset_global_routing_layer_adjustment"
                new_file += " $::env(MIN_ROUTING_LAYER)"
                new_file += "-$::env(MAX_ROUTING_LAYER)"
                new_file += f" {value}"
            elif re.search(f"{layer_cmd}.*{layer}", new_file):
                new_file = re.sub(
                    f"({layer_cmd}.*{layer}).*\n(.*)", f"\\1 {value}\n\\2", new_file
                )
            else:
                new_file += f"\n{layer_cmd} {layer} {value}\n"
        elif key == "GR_SEED":
            new_file += f"\nset_global_routing_random -seed {value}\n"
        else:
            print(
                f"[WARN TUN-0028] {key} variable not supported in context of FastRoute TCL files"
            )
            continue

    with open(filepath, "w") as file:
        file.write(new_file)


def parse_flow_variables(base_dir, platform):
    """
    Parse the flow variables from source
    - Code: Makefile `vars` target output

    TODO: Tests.

    Output:
    - flow_variables: set of flow variables
    """
    # first, generate vars.tcl
    makefile_path = os.path.join(base_dir, "flow")
    result = subprocess.run(
        ["make", "-C", makefile_path, "vars", f"PLATFORM={platform}"],
        capture_output=True,
    )
    if result.returncode != 0:
        print(f"[ERROR TUN-0018] Makefile failed with error code {result.returncode}.")
        sys.exit(1)
    if not os.path.exists(os.path.join(makefile_path, "vars.tcl")):
        print("[ERROR TUN-0019] Makefile did not generate vars.tcl.")
        sys.exit(1)

    # for code parsing, you need to parse from both scripts and vars.tcl file.
    pattern = r"(?:::)?env\((.*?)\)"
    files = glob.glob(os.path.join(makefile_path, "scripts/*.tcl"))
    files.append(os.path.join(makefile_path, "vars.tcl"))
    variables = set()
    for file in files:
        with open(file) as fp:
            matches = re.findall(pattern, fp.read())
        for match in matches:
            for variable in match.split("\n"):
                variables.add(variable.strip().upper())
    return variables


def parse_tunable_variables():
    """
    Parse the tunable variables from variables.yaml
    TODO: Tests.
    """
    cur_path = os.path.dirname(os.path.realpath(__file__))
    vars_path = os.path.join(cur_path, "../../../../flow/scripts/variables.yaml")

    # Read from variables.yaml and get variables with tunable = 1
    with open(vars_path) as file:
        result = yaml.safe_load(file)
    variables = {key for key, value in result.items() if value.get("tunable", 0) == 1}
    return variables


def parse_config(
    files,
    config,
    base_dir,
    platform,
    sdc_original,
    fr_original,
    path=os.getcwd(),
):
    """
    Parse configuration received from tune into make variables.
    """
    options = ""
    sdc = {}
    fast_route = {}
    flow_variables = parse_tunable_variables()
    for key, value in config.items():
        # Keys that begin with underscore need special handling.
        if key.startswith("_"):
            # Variables to be injected into fastroute.tcl
            if key.startswith("_FR_"):
                fast_route[key[4:]] = value
            # Variables to be injected into constraints.sdc
            elif key.startswith("_SDC_"):
                sdc[key[5:]] = value
            # Special substitution cases
            elif key == "_PINS_DISTANCE":
                options += f' PLACE_PINS_ARGS="-min_distance {value}"'
            elif key == "_SYNTH_FLATTEN":
                print(
                    "[WARNING TUN-0013] Non-flatten the designs are not "
                    "fully supported, ignoring _SYNTH_FLATTEN parameter."
                )
        # Default case is VAR=VALUE
        else:
            # Sanity check: ignore all flow variables that are not tunable
            if key not in flow_variables:
                print(f"[ERROR TUN-0017] Variable {key} is not tunable.")
                sys.exit(1)
            options += f" {key}={value}"
    if sdc:
        write_sdc(sdc, sdc_original, files["_SDC_FILE_PATH"])
        options += f" SDC_FILE={files['_SDC_FILE_PATH']}"
    if fast_route:
        write_fast_route(fast_route, platform, fr_original, files["_FR_FILE_PATH"])
        options += f" FASTROUTE_TCL={files['_FR_FILE_PATH']}"
    return options


def run_command(
    args, cmd, timeout=None, stderr_file=None, stdout_file=None, fail_fast=False
):
    """
    Wrapper for subprocess.run
    Allows to run shell command, control print and exceptions.
    """
    process = subprocess.run(
        cmd, timeout=timeout, capture_output=True, text=True, check=False, shell=True
    )
    if stderr_file is not None and process.stderr != "":
        with open(stderr_file, "a") as file:
            file.write(f"\n\n{cmd}\n{process.stderr}")
    if stdout_file is not None and process.stdout != "":
        with open(stdout_file, "a") as file:
            file.write(f"\n\n{cmd}\n{process.stdout}")
    if args.verbose >= 1:
        print(process.stderr)
    if args.verbose >= 2:
        print(process.stdout)

    if fail_fast and process.returncode != 0:
        raise RuntimeError


def openroad(
    args,
    base_dir,
    parameters,
    flow_variant,
    path="",
    install_path=None,
):
    """
    Run OpenROAD-flow-scripts with a given set of parameters.
    """
    # Make sure path ends in a slash, i.e., is a folder
    flow_variant = f"{args.experiment}/{flow_variant}"
    if path != "":
        log_path = f"{path}/{flow_variant}/"
        report_path = log_path.replace("logs", "reports")
        run_command(args, f"mkdir -p {log_path}")
        run_command(args, f"mkdir -p {report_path}")
    else:
        log_path = report_path = os.getcwd() + "/"

    if install_path is None:
        install_path = os.path.join(base_dir, "tools/install")

    export_command = f"export PATH={install_path}/OpenROAD/bin"
    export_command += f":{install_path}/yosys/bin:$PATH"
    export_command += " && "

    make_command = export_command
    make_command += f"make"
    if args.stage_stop != "all":
        for i in range (ALLOWED_OPENROAD_STAGES.index(args.stage_stop) + 1):
            make_command += f" {ALLOWED_OPENROAD_STAGES[i]}" # Append preceding flow stages to make command
    make_command += f" -C {base_dir}/flow DESIGN_CONFIG=designs/"
    make_command += f"{args.platform}/{args.design}/config.mk"
    make_command += f" PLATFORM={args.platform}"
    make_command += f" FLOW_VARIANT={args.experiment}/{flow_variant} {parameters}"
    make_command += f" EQUIVALENCE_CHECK=0"
    make_command += f" NUM_CORES={args.openroad_threads} SHELL=bash"
    run_command(args, f"cd {base_dir}")

    print(f"TR make_command: {make_command}")

    run_command(
        args,
        make_command,
        timeout=args.timeout,
        stderr_file=f"{log_path}error-make-finish.log",
        stdout_file=f"{log_path}make-finish-stdout.log",
    )

    metrics_file = os.path.abspath(os.path.join(report_path, "metrics.json"))
    metrics_command = export_command
    metrics_command += f"{base_dir}/flow/util/genMetrics.py -x"
    metrics_command += f" -v {flow_variant}"
    metrics_command += f" -d {args.design}"
    metrics_command += f" -p {args.platform}"
    metrics_command += f" -o {metrics_file}"
    run_command(
        args,
        metrics_command,
        stderr_file=f"{log_path}error-metrics.log",
        stdout_file=f"{log_path}metrics-stdout.log",
    )

    return metrics_file


def read_metrics(file_name):
    """
    Collects metrics to evaluate the user-defined objective function.
    """
    with open(file_name) as file:
        data = json.load(file)
    clk_period = 9999999
    worst_slack = "ERR"
    wirelength = "ERR"
    num_drc = "ERR"
    total_power = "ERR"
    core_util = "ERR"
    final_util = "ERR"
    design_area = "ERR"
    die_area = "ERR"
    core_area = "ERR"
    for stage_name, value in data.items():
        if stage_name == "constraints" and len(value["clocks__details"]) > 0:
            clk_period = float(value["clocks__details"][0].split()[1])
        if stage_name == "floorplan" and "design__instance__utilization" in value:
            core_util = value["design__instance__utilization"]
        if stage_name == "detailedroute" and "route__drc_errors" in value:
            num_drc = value["route__drc_errors"]
        if stage_name == "detailedroute" and "route__wirelength" in value:
            wirelength = value["route__wirelength"]
        if stage_name == "finish" and "timing__setup__ws" in value:
            worst_slack = value["timing__setup__ws"]
        if stage_name == "finish" and "power__total" in value:
            total_power = value["power__total"]
        if stage_name == "finish" and "design__instance__utilization" in value:
            final_util = value["design__instance__utilization"]
        if stage_name == "finish" and "design__instance__area" in value:
            design_area = value["design__instance__area"]
        if stage_name == "finish" and "design__core__area" in value:
            core_area = value["design__core__area"]
        if stage_name == "finish" and "design__die__area" in value:
            die_area = value["design__die__area"]
    ret = {
        "clk_period": clk_period,
        "worst_slack": worst_slack,
        "total_power": total_power,
        "core_util": core_util,
        "final_util": final_util,
        "design_area": design_area,
        "core_area": core_area,
        "die_area": die_area,
        "wirelength": wirelength,
        "num_drc": num_drc,
    }
    return ret


def read_config(args, orfs_flow_dir):
    """
    Please consider inclusive, exclusive
    Most type uses [min, max)
    But, Quantization makes the upper bound inclusive.
    e.g., qrandint and qlograndint uses [min, max]
    step value is used for quantized type (e.g., quniform). Otherwise, write 0.
    When min==max, it means the constant value
    """

    file_name = os.path.abspath(args.config)

    def validate_power_of_2_range(min_val, max_val):
        """Validates that min and max values are powers of 2"""
        def is_power_of_2(n):
            return n > 0 and (n & (n - 1)) == 0

        if not is_power_of_2(min_val) or not is_power_of_2(max_val):
            print(f"[ERROR TUN-00xx] When step is 'power_of_2', min ({min_val}) and max ({max_val}) must be powers of 2.") # TODO: Add parameter name to error message
            sys.exit(1)

    def generate_power_of_2_sequence(min_val, max_val):
        """Generates sequence of powers of 2 between min and max inclusive"""
        sequence = []
        current = min_val
        while current <= max_val:
            sequence.append(current)
            current *= 2
        return sequence

    def apply_condition(config, data):
        # TODO: tune.sample_from only supports random search algorithm.
        # To make conditional parameter for the other algorithms, different
        # algorithms should take different methods (will be added)
        if args.algorithm != "random":
            return config
        dp_pad_min = data["CELL_PAD_IN_SITES_DETAIL_PLACEMENT"]["minmax"][0]
        dp_pad_step = data["CELL_PAD_IN_SITES_DETAIL_PLACEMENT"]["step"]
        if dp_pad_step == 1:
            config["CELL_PAD_IN_SITES_DETAIL_PLACEMENT"] = tune.sample_from(
                lambda spec: np.random.randint(
                    dp_pad_min, spec.config.CELL_PAD_IN_SITES_GLOBAL_PLACEMENT + 1
                )
            )
        if dp_pad_step > 1:
            config["CELL_PAD_IN_SITES_DETAIL_PLACEMENT"] = tune.sample_from(
                lambda spec: random.randrange(
                    dp_pad_min,
                    spec.config.CELL_PAD_IN_SITES_GLOBAL_PLACEMENT + 1,
                    dp_pad_step,
                )
            )
        return config

    def read(path):
        # if file path does not exist, return empty string
        print(os.path.abspath(path))
        if not os.path.isfile(os.path.abspath(path)):
            return ""
        with open(os.path.abspath(path), "r") as file:
            ret = file.read()
        return ret

    def read_tune(this):
        if this["type"] == "choice":
            return tune.choice(this["values"])
        else:
            min_, max_ = this["minmax"]
            if min_ == max_:
                # Returning a choice of a single element allow pbt algorithm to
                # work. pbt does not accept single values as tunable.
                return tune.choice([min_, max_])

            # Handle power_of_2 step
            if isinstance(this["step"], str):
                if this["step"] == "power_of_2":
                    validate_power_of_2_range(min_, max_)
                    sequence = generate_power_of_2_sequence(min_, max_)
                    return tune.choice(sequence)
                else:
                    print(f"[ERROR TUN-00xx] Invalid step value '{this['step']}.") # TODO: Add parameter name to error message
                    sys.exit(1)

            if this["type"] == "int":
                if isinstance(this["step"], str) and this["step"] == "power_of_2":
                    validate_power_of_2_range(min_, max_)
                    sequence = generate_power_of_2_sequence(min_, max_)
                    return tune.choice(sequence)
                elif this["step"] == 1:
                    return tune.randint(min_, max_)
                return tune.choice(np.ndarray.tolist(np.arange(min_, max_, this["step"])))
            elif this["type"] == "float":
                if this["step"] == 0:
                    return tune.uniform(min_, max_)
                return tune.choice(np.ndarray.tolist(np.arange(min_, max_, this["step"])))
        return None

    def read_tune_ax(name, this):
        """
        Ax format: https://ax.dev/versions/0.3.7/api/service.html
        """
        dict_ = dict(name=name)

        if this["type"] == "choice":
            dict_["type"] = "choice"
            dict_["values"] = this["values"]
            dict_["is_ordered"] = False
            dict_["sort_values"] = False
            if isinstance(this["values"][0], str):
                dict_["value_type"] = "str"
            elif isinstance(this["values"][0], int):
                dict_["value_type"] = "int"
            elif isinstance(this["values"][0], float):
                dict_["value_type"] = "float"
            return dict_

        if "minmax" not in this:
            return None
        min_, max_ = this["minmax"]
        if min_ == max_:
            dict_["type"] = "fixed"
            dict_["value"] = min_
        elif this["type"] == "int":
            if isinstance(this["step"], str) and this["step"] == "power_of_2":
                validate_power_of_2_range(min_, max_)
                dict_["type"] = "choice"
                dict_["values"] = generate_power_of_2_sequence(min_, max_)
                dict_["is_ordered"] = True
                dict_["sort_values"] = False
            elif this["step"] == 1:
                dict_["type"] = "range"
                dict_["bounds"] = [min_, max_]
            else:
                dict_["type"] = "choice"
                dict_["values"] = np.arange(min_, max_, this["step"]).tolist()
                dict_["is_ordered"] = True
                dict_["sort_values"] = False
            dict_["value_type"] = "int"
        elif this["type"] == "float":
            if this["step"] == 1:
                dict_["type"] = "choice"
                dict_["values"] = np.arange(min_, max_, this["step"]).tolist()
                dict_["is_ordered"] = True
                dict_["sort_values"] = False
            else:
                dict_["type"] = "range"
                dict_["bounds"] = [min_, max_]
            dict_["value_type"] = "float"
        return dict_

    def parse_filepaths(json_config):
        """
        Parse file paths from JSON configuration file.
        """
        if "files" not in json_config.keys():
            print(f"[ERROR TUN-0020] Files group is missing in JSON configuration file.")
            sys.exit(1)

        if "_SDC_FILE_PATH" not in json_config["files"].keys() or json_config["files"]["_SDC_FILE_PATH"] == "":
            print(f"[ERROR TUN-0020] SDC file (key '_SDC_FILE_PATH') is missing in JSON configuration file.")
            sys.exit(1)

        if "_FR_FILE_PATH" not in json_config["files"].keys() or json_config["files"]["_FR_FILE_PATH"] == "":
            print(f"[ERROR TUN-0020] FR file (key '_FR_FILE_PATH') is missing in JSON configuration file.")
            sys.exit(1)

        if args.run_sim and ("_SIM_FILE_PATH" not in json_config["files"].keys() or json_config["files"]["_SIM_FILE_PATH"] == ""):
            print(f"[ERROR TUN-0020] Simulation file (key '_SIM_FILE_PATH') is missing in JSON configuration file.")
            sys.exit(1)

        sdc_file = ""
        fr_file = ""
        for key, filepath in json_config["files"].items():
            if "_FILE_PATH" not in key:
                print(f"[WARNING TUN-xxx] Field {key} isn't valid for group 'files'. Ignoring it.")
                continue

            if key in JSON_FILEPATHS.keys() and JSON_FILEPATHS[key] is not None:
                print(f"[WARNING TUN-0004] Obtained more than one file path for {key}.")

            if filepath == "":
                print(f"[WARNING TUN-xxx] Value for key {key} in the 'files' group is blank. Ignoring it.")
                continue

            full_path = os.path.join(os.path.dirname(file_name), filepath)
            JSON_FILEPATHS[key] = str(full_path).partition("flow/")[2]

            if key == "_SDC_FILE_PATH":
                if sdc_file != "":
                    print("[WARNING TUN-0004] Overwriting SDC base file.")
                sdc_file = read(orfs_flow_dir + f"/{JSON_FILEPATHS['_SDC_FILE_PATH']}")
            elif key == "_FR_FILE_PATH":
                if fr_file != "":
                    print("[WARNING TUN-0005] Overwriting FastRoute base file.")
                fr_file = read(orfs_flow_dir + f"/{JSON_FILEPATHS['_FR_FILE_PATH']}")

        return sdc_file, fr_file

    def parse_metrics(json_config):
        """
        Parse metrics from JSON configuration file.
        """
        if "score_metrics" not in json_config.keys():
            print(f"[ERROR TUN-0020] Metrics group is missing in JSON configuration file.")
            sys.exit(1)

        for metric_name, metric_config in json_config["score_metrics"].items():
            if metric_config is None:
                print(f"[WARNING TUN-0005] Metric {metric_name} has no configuration. Ignoring it.")
                continue
            METRICS_CONFIG[metric_name] = metric_config

    def parse_parameters(json_config):
        """
        Parse parameters from JSON configuration file.
        """

        if "parameters" not in json_config.keys():
            print(f"[ERROR TUN-0020] Parameters group is missing in JSON configuration file.")
            sys.exit(1)

        if args.mode == "tune" and args.algorithm == "ax":
            config = list()
        else:
            config = dict()

        top_param_or_def_found, package_param_or_def_found = False, False
        for key, value in json_config["parameters"].items():
            if key == "best_result":
                continue

            # If we're not running any OpenROAD stage, ignore all non-RTL parameters
            if args.stage_stop == "none" and not (key.startswith("_TOP") or key.startswith("_PACKAGE")):
                continue

            top_param_or_def_found = ("_TOP_PARAM" in key) or top_param_or_def_found
            package_param_or_def_found = ("_PACKAGE" in key) or package_param_or_def_found

            if not isinstance(value, dict):
                if args.mode == "tune" and args.algorithm == "ax":
                    param_dict = read_tune_ax(key, value)
                    if param_dict:
                        config.append(param_dict)
                elif args.mode == "tune" and args.algorithm == "pbt":
                    param_dict = read_tune(value)
                    if param_dict:
                        config[key] = param_dict
                else:
                    config[key] = value
            elif args.mode == "sweep":
                config[key] = value
            elif args.mode == "tune" and args.algorithm == "ax":
                config.append(read_tune_ax(key, value))
            elif args.mode == "tune" and args.algorithm == "pbt":
                config[key] = read_tune(value)
            elif args.mode == "tune":
                config[key] = read_tune(value)

        if args.mode == "tune":
            config = apply_condition(config, json_config["parameters"])

        if top_param_or_def_found and JSON_FILEPATHS["_TOP_LEVEL_FILE_PATH"] == None:
            print(f"[ERROR TUN-0020] _TOP_PARAM_ or _TOP_DEF_ found in JSON configuration file but _TOP_LEVEL_FILE_PATH is missing.")
            sys.exit(1)
        if package_param_or_def_found and JSON_FILEPATHS["_PACKAGE_FILE_PATH"] == None:
            print(f"[ERROR TUN-0020] _PACKAGE_PARAM_ or _PACKAGE_DEF_ found in JSON configuration file but _PACKAGE_FILE_PATH is missing.")
            sys.exit(1)

        return config

    # Check file exists and whether it is a valid JSON file.
    assert os.path.isfile(file_name), f"File {file_name} not found."
    try:
        with open(file_name) as file:
            data = json.load(file)
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON file: {file_name}")

    sdc_original, fr_original = parse_filepaths(data)
    parse_metrics(data)
    config = parse_parameters(data)

    return config, JSON_FILEPATHS, sdc_original, fr_original


def clone(args, path):
    """
    Clone base repo in the remote machine. Only used for Kubernetes at GCP.
    """
    if args.git_clone:
        run_command(args, f"rm -rf {path}")
    if not os.path.isdir(f"{path}/.git"):
        git_command = "git clone --depth 1 --recursive --single-branch"
        git_command += f" {args.git_clone_args}"
        git_command += f" --branch {args.git_orfs_branch}"
        git_command += f" {args.git_url} {path}"
        run_command(args, git_command)


def build(args, base, install):
    """
    Build OpenROAD, Yosys and other dependencies.
    """
    build_command = f'cd "{base}"'
    if args.git_clean:
        build_command += " && git clean -xdf tools"
        build_command += " && git submodule foreach --recursive git clean -xdf"
    if (
        args.git_clean
        or not os.path.isfile(f"{install}/OpenROAD/bin/openroad")
        or not os.path.isfile(f"{install}/yosys/bin/yosys")
    ):
        build_command += ' && bash -ic "./build_openroad.sh'
        # Some GCP machines have 200+ cores. Let's be reasonable...
        build_command += f" --local --nice --threads {min(32, cpu_count())}"
        if args.git_latest:
            build_command += " --latest"
        build_command += f' {args.build_args}"'
    run_command(args, build_command)


def copy_directory(repo_dir, copy_dir, args):
    """
    Makes a local copy of the repo, but discards unnecessary directories (unused platforms and designs, etc).
    This gives each run a fresh copy of the design file to avoid race conditions.
    """

    dont_copy = ["logs", "reports", "results", "objects", "docs", "autotuner_env"]
    for pattern, directory in [(args.platform, "platforms"), (args.platform, "designs"), (args.design, "designs/src")]:
        target_dir = ORFS_FLOW_DIR + f"/{directory}"
        all_patterns = [d for d in os.listdir(target_dir)
                        if os.path.isdir(os.path.join(target_dir, d))]
        other_patterns = [p for p in all_patterns if (p != pattern and p != "src" and p != "common")]
        dont_copy.extend(other_patterns)

    copytree(str(repo_dir), str(copy_dir), ignore=ignore_patterns(*dont_copy))


@ray.remote
def setup_repo(args, base):
    """
    Clone ORFS repository and compile binaries.
    """
    print(f"[INFO TUN-0000] Remote folder: {base}")
    install = f"{base}/tools/install"
    if args.server is not None:
        clone(base)
    build(base, install)
    return install


def prepare_ray_server(args):
    """
    Prepares Ray server and returns basic directories.
    # TR TODO: Handle ORFS_FLOW_DIR (comes from global in utils.py (and imported in distributed.py or returned from here?))
    """
    # Connect to remote Ray server if any, otherwise will run locally
    if args.server is not None:
        # At GCP we have a NFS folder that is present for all worker nodes.
        # This allows to build required binaries once. We clone, build and
        # store intermediate files at LOCAL_DIR.
        with open(args.config) as config_file:
            local_dir = "/shared-data/autotuner"
            local_dir += f"-orfs-{args.git_orfs_branch}"
            if args.git_or_branch != "":
                local_dir += f"-or-{args.git_or_branch}"
            if args.git_latest:
                local_dir += "-or-latest"
        # Connect to ray server before first remote execution.
        ray.init(f"ray://{args.server}:{args.port}")
        # Remote functions return a task id and are non-blocking. Since we
        # need the setup repo before continuing, we call ray.get() to wait
        # for its completion.
        install_path = ray.get(setup_repo.remote(local_dir))
        orfs_flow_dir = os.path.join(local_dir, "flow")
        local_dir += f"/flow/logs/{args.platform}/{args.design}"
        print("[INFO TUN-0001] NFS setup completed.")
    else:
        orfs_dir = getattr(args, "orfs", None)
        # For local runs, use the same folder as other ORFS utilities.
        orfs_flow_dir = os.path.abspath(
            os.path.join(orfs_dir, "flow")
            if orfs_dir
            else os.path.join(os.path.dirname(__file__), "../../../../flow")
        )
        
        local_dir = f"logs/{args.platform}/{args.design}"
        local_dir = os.path.join(orfs_flow_dir, local_dir)
        install_path = os.path.abspath(os.path.join(orfs_flow_dir, "../tools/install"))
        install_path = os.path.abspath("foss/tools") # TODO: TR OVERRIDE for oseda...think about what to do about this
    return local_dir, orfs_flow_dir, install_path


@ray.remote
def openroad_distributed(
    args,
    repo_dir,
    config,
    path,
    sdc_original,
    fr_original,
    install_path,
    variant=None,
):
    """Simple wrapper to run openroad distributed with Ray."""
    config = parse_config(
        copy_dir,
        config=config,
        base_dir=repo_dir,
        platform=args.platform,
        sdc_original=sdc_original,
        fr_original=fr_original,
    )
    if variant is None:
        variant = config.replace(" ", "_").replace("=", "_")
    t = time()
    metric_file = openroad(
        args=args,
        base_dir=repo_dir,
        parameters=config,
        flow_variant=f"{uuid()}-{variant}",
        path=path,
        install_path=install_path,
    )
    duration = time() - t
    return metric_file, duration


@ray.remote
def consumer(queue):
    """consumer"""
    while not queue.empty():
        next_item = queue.get()
        name = next_item[1]
        print(f"[INFO TUN-0007] Scheduling run for parameter {name}.")
        ray.get(openroad_distributed.remote(*next_item))
        print(f"[INFO TUN-0008] Finished run for parameter {name}.")
