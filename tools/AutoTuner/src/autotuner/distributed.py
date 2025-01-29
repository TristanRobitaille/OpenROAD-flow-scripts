"""
This scripts handles sweeping and tuning of OpenROAD-flow-scripts parameters.
Dependencies are documented in pip format at distributed-requirements.txt

For both sweep and tune modes:
    python3 distributed.py -h

Note: the order of the parameters matter.
Arguments --design, --platform, --config, --stage_stop are always required and should
precede the <mode>.

AutoTuner:
    python3 distributed.py tune -h
    python3 distributed.py --design gcd --platform sky130hd --stage_stop all \
                           --config ../designs/sky130hd/gcd/autotuner.json \
                           tune
    Example:

Parameter sweeping:
    python3 distributed.py sweep -h
    Example:
    python3 distributed.py --design gcd --platform sky130hd --stage_stop all \
                           --config distributed-sweep-example.json \
                           sweep
"""

import argparse
import json
import os
import shutil
import re
import sys
import glob
import subprocess
import random
from pathlib import Path
from copy import deepcopy
from datetime import datetime
from multiprocessing import cpu_count
from subprocess import run
from itertools import product
from collections import namedtuple
from uuid import uuid4 as uuid

import numpy as np
import torch

import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.ax import AxSearch
from ray.tune.search.basic_variant import BasicVariantGenerator
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search.optuna import OptunaSearch
from ray.util.queue import Queue

from ax.service.ax_client import AxClient

from VerilogRewriter import VerilogRewriter

DATE = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
ORFS_URL = Path("https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts")
JSON_FILES_BASE = { # Base path for files defined in the .json file
    "_SDC_FILE_PATH": None,
    "_FR_FILE_PATH": None,
    "_PACKAGE_FILE_PATH": None,
    "_TOP_LEVEL_FILE_PATH": None,
}
METRICS_CONFIG = dict() # Configuration for the metrics used to compute the score for each run
METRIC = "minimum"
ERROR_METRIC = 9e99
# ORFS_FLOW_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../flow"))
ORFS_FLOW_DIR = Path(__file__).resolve().parent / "../../../../flow"
ORFS_FLOW_DIR = ORFS_FLOW_DIR.resolve()
ALLOWED_SIM_STAGES = ["rtl_sim"]
ALLOWED_OPENROAD_STAGES = ["floorplan", "place", "cts", "route", "all"]
FLOW_METRICS_MAP = {
    "floorplan": {  "total_power"   : ["floorplan", "power__total"],
                    "core_util"     : ["floorplan", "design__instance__utilization"],
                    "worst_slack"   : ["floorplan", "timing__setup__ws"]},

    "place": {      "total_power"   : ["detailedplace", "power__total"],
                    "wirelength"    : ["detailedplace", "route__wirelength__estimated"],
                    "core_util"     : ["detailedplace", "design__instance__utilization"],
                    "worst_slack"   : ["detailedplace", "timing__setup__ws"]},

    "cts": {        "total_power"   : ["cts", "power__total"],
                    "wirelength"    : ["cts", "route__wirelength__estimated"],
                    "core_util"     : ["cts", "design__instance__utilization"],
                    "worst_slack"   : ["cts", "timing__setup__ws"]},

    "route": {      "total_power"   : ["globalroute","power__total"],
                    "wirelength"    : ["detailedroute","route__wirelength"],
                    "core_util"     : ["globalroute", "design__instance__utilization"],
                    "num_drc"       : ["detailedroute","route__drc_errors"],
                    "worst_slack"   : ["globalroute", "timing__setup__ws"]},

    "all": {        "total_power"   : ["finish", "power__total"],
                    "wirelength"    : ["detailedroute","route__wirelength"],
                    "core_util"     : ["globalroute", "design__instance__utilization"],
                    "num_drc"       : ["detailedroute","route__drc_errors"],
                    "worst_slack"   : ["finish", "timing__setup__ws"]}
}

from enum import Enum
# ----- ENUMS ----- #
class TerminalTool(Enum):
    warning = "\033[93mWARNING\033[0m"
    error = "\033[91mERROR\033[0m"

# ----- CLASSES ----- #
class AutoTunerBase(tune.Trainable):
    """
    AutoTuner base class for experiments.
    """

    def setup(self, config):
        """
        Initializes the experimental setup for the AutoTunerBase class.

        Prepares the environment by creating a directory structure,
        copying the repository (excluding certain directories), and
        setting up configuration parameters for the experiment.

        Parameters:
        - config (dict): Hyperparameters, parameters and defines for this run.
        """

        # We create the following directory structure:
        #      1/     2/         3/       4/                5/   6/
        # <repo>/<logs>/<platform>/<design>/<experiment>-DATE/<id>/<cwd>
        repo_dir = Path(__file__).resolve().parent / "../../../.."
        self.repo_dir = repo_dir.resolve()
        self.step_ = 0
        self.variant = f"variant-{self.__class__.__name__}-{self.trial_id}-or"
        self.copy_dir = Path.cwd() / self.variant

        self.files = copy_repo(self.repo_dir, self.copy_dir)

        self.parameters = parse_config(config, self.files)

    def step(self):
        """
        Run step experiment and compute its score.
        """
        metrics_file = ""
        if args.stage_stop in ALLOWED_SIM_STAGES:
            metrics_file = run_rtl_sim(self.copy_dir, self.variant, self.files)
        else:
            metrics_file = openroad(self.copy_dir, self.parameters, self.variant)
        self.step_ += 1
        score = self.evaluate(self.read_metrics(metrics_file))
        # Feed the score back to Tune. return must match 'metric' used in tune.run()
        return {METRIC: score}

    def evaluate(self, metrics):
        """
        User-defined evaluation function.
        It can change in any form to minimize the score (return value).
        Default evaluation function optimizes effective clock period.
        """

        if args.stage_stop in ALLOWED_OPENROAD_STAGES:
            if metrics["clk_period"] == "N/A" or metrics["worst_slack"] == "N/A":
                return ERROR_METRIC

            gamma = (metrics["clk_period"] - metrics["worst_slack"]) / 10
            score = metrics["clk_period"] - metrics["worst_slack"]
            score = score * (self.step_ / 100) ** (-1)

            if args.stage_stop == "route" or args.stage_stop == "all": # DRC errors are only available in route stage
                score += gamma * metrics["num_drc"]
        else: # RTL sim
            score = 0
            for metric_name, metric_config in METRICS_CONFIG.items():
                if metric_name in metrics:
                    score += metric_config["coeff"] * metrics[metric_name]
        return score

    @classmethod
    def read_metrics(cls, file_name):
        """
        Collects metrics to evaluate the user-defined objective function.
        """
        with open(file_name) as file:
            data = json.load(file)

        if args.stage_stop in ALLOWED_OPENROAD_STAGES:
            metrics_dict = {
                "clk_period": 9999999,
                "worst_slack": "N/A",
                "wirelength": "N/A",
                "num_drc": "N/A",
                "total_power": "N/A",
                "core_util": "N/A",
            }
            if len(data["constraints"]["clocks__details"]) > 0:
                metrics_dict["clk_period"] = float(data["constraints"]["clocks__details"][0].split()[1])
            for metric_name in metrics_dict.keys():
                if metric_name in FLOW_METRICS_MAP[args.stage_stop].keys():
                    stage_name = FLOW_METRICS_MAP[args.stage_stop][metric_name][0]
                    metric_json_name = FLOW_METRICS_MAP[args.stage_stop][metric_name][1]
                    metrics_dict[metric_name] = data[stage_name][metric_json_name]
        else:
            metrics_dict = dict()
            for metric_name, value in data[args.stage_stop].items():
                metrics_dict[metric_name] = value
    
        return metrics_dict


class ScoreImprov(AutoTunerBase):
    """
    Extends AutoTunerBase to improve score metrics.

    This class overrides the evaluation method to compute a score.
    It uses a reference configuration to compare and compute the improvements.
    Users are invited to modify the get_score function to suit their specific needs.
    """

    @classmethod
    def get_score(cls, metrics):
        """
        Compute score term for evaluate.
        """

        def percent(x_1, x_2):
            return 100 * (x_1 - x_2) / x_1

        def sum_metrics(collected_metrics):
            """
            Compute the sum of all metrics * coefficient.
            Lower values of score are better.
            """
            score_upper_bound, score = 0, 0
            for metric_name, metric_data in METRICS_CONFIG.items():
                score_upper_bound += (100 * metric_data["coeff"])
                score += (metric_data["coeff"] * collected_metrics[metric_name])
            
            return score_upper_bound, score

        if args.stage_stop in ALLOWED_OPENROAD_STAGES:
            assert (metrics["clk_period"] != "N/A" and metrics["worst_slack"] != "N/A")
            assert (metrics["total_power"] != "N/A" and metrics["total_power"] != "N/A")
            assert (metrics["core_util"] != "N/A" and metrics["core_util"] != "N/A")

            eff_clk_period = metrics["clk_period"]
            if metrics["worst_slack"] < 0:
                eff_clk_period -= metrics["worst_slack"]

            eff_clk_period_ref = reference["clk_period"]
            if reference["worst_slack"] < 0:
                eff_clk_period_ref -= reference["worst_slack"]

            collected_metrics = {
                "performance": percent(eff_clk_period_ref, eff_clk_period),
                "power": percent(reference["total_power"], metrics["total_power"]),
                "area": percent(100 - reference["core_util"], 100 - metrics["core_util"]),
            }

            score_upper_bound, score = sum_metrics(collected_metrics)
            return score
        else: # RTL sim
            collected_metrics = dict()
            for metric_name, metric_val in metrics.items(): # This assumes that for all metrics, lower is better
                if metric_name not in reference:
                    print(f"[{TerminalTool.warning}] Metric {metric_name} not found in reference! Exiting.")
                    sys.exit(1)

                collected_metrics[metric_name] = percent(reference[metric_name], metric_val)
            score_upper_bound, score = sum_metrics(collected_metrics)
            return score


    def evaluate(self, metrics):
        if args.stage_stop in ALLOWED_OPENROAD_STAGES:
            for metric in FLOW_METRICS_MAP[args.stage_stop].keys():
                if metrics[metric] == "N/A" or reference[metric] == "N/A":
                    return ERROR_METRIC

            score = self.get_score(metrics)
            score_normalized = score * (self.step_ / 100) ** (-1)
            if args.stage_stop == "route" or args.stage_stop == "all": # DRC errors are only available in route stage
                gamma = score / 10
                score_normalized += gamma * metrics["num_drc"]
        else: # RTL sim
            score = self.get_score(metrics)
            score_normalized = score * (self.step_ / 100) ** (-1)
        return score


def copy_repo(repo_dir, copy_dir):
    """
    Makes a local copy of the repo, but discards unnecessary directories (unused platforms and designs, etc).
    This gives each run a fresh copy of the design file to avoid race conditions.
    """
    # Make list of patterns to ignore when copying the repo to reduce the size of the copy
    dont_copy = ["logs", "reports", "results", "objects", "docs"]
    for pattern, directory in [(args.platform, "platforms"), (args.platform, "designs"), (args.design, "designs/src")]:
        target_dir = ORFS_FLOW_DIR / directory
        all_patterns = [d for d in os.listdir(target_dir)
                        if os.path.isdir(os.path.join(target_dir, d))]
        other_patterns = [p for p in all_patterns if (p != pattern and p != "src" and p != "common")]
        dont_copy.extend(other_patterns)

    shutil.copytree(str(repo_dir), str(copy_dir), ignore=shutil.ignore_patterns(*dont_copy))

    # Update the file paths in the configuration dictionary
    files = deepcopy(JSON_FILES_BASE)
    for key, filepath in files.items():
        if filepath is not None:
            files[key] = copy_dir / filepath

    return files


def read_config(file_name):
    """
    Please consider inclusive, exclusive
    Most type uses [min, max)
    But, Quantization makes the upper bound inclusive.
    e.g., qrandint and qlograndint uses [min, max]
    step value is used for quantized type (e.g., quniform). Otherwise, write 0.
    When min==max, it means the constant value
    """

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

    def read_tune(this):
        if this["type"] == "choice":
            return tune.choice(this["values"])
        else:
            min_, max_ = this["minmax"]
            if min_ == max_:
                # Returning a choice of a single element allow pbt algorithm to
                # work. pbt does not accept single values as tunable.
                return tune.choice([min_, max_])
            if this["type"] == "int":
                if this["step"] == 1:
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
            if this["step"] == 1:
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

        if "_SDC_FILE_PATH" not in json_config["files"].keys():
            print(f"[ERROR TUN-0020] SDC file (key '_SDC_FILE_PATH') is missing in JSON configuration file.")
            sys.exit(1)

        if "_FR_FILE_PATH" not in json_config["files"].keys():
            print(f"[ERROR TUN-0020] FR file (key '_FR_FILE_PATH') is missing in JSON configuration file.")
            sys.exit(1)

        if args.stage_stop in ALLOWED_SIM_STAGES and "_SIM_FILE_PATH" not in json_config["files"].keys():
            print(f"[ERROR TUN-0020] Simulation file (key '_SIM_FILE_PATH') is missing in JSON configuration file.")
            sys.exit(1)

        for key, filepath in json_config["files"].items():
            if "_FILE_PATH" not in key:
                print(f"[WARNING TUN-xxx] Field {key} isn't valid for group 'files'. Ignoring it.")
                continue

            if key in JSON_FILES_BASE.keys() and JSON_FILES_BASE[key] is not None:
                print(f"[WARNING TUN-0004] Obtained more than one file path for {key}.")

            if filepath == "":
                print(f"[WARNING TUN-xxx] Value for key {key} in the 'files' group is blank. Ignoring it.")
                continue

            filepath = Path(filepath)

            base_dir = Path(file_name).parent
            full_path = base_dir / Path(filepath)
            JSON_FILES_BASE[key] = Path(str(full_path).partition("OpenROAD-flow-scripts/")[2])

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

            # For RTL simulation, only consider parameters starting with _TOP or _PACKAGE
            if args.stage_stop in ALLOWED_SIM_STAGES and not (key.startswith("_TOP") or key.startswith("_PACKAGE")):
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

        if top_param_or_def_found and JSON_FILES_BASE["_TOP_LEVEL_FILE_PATH"] == None:
            print(f"[ERROR TUN-0020] _TOP_PARAM_ or _TOP_DEF_ found in JSON configuration file but _TOP_LEVEL_FILE_PATH is missing.")
            sys.exit(1)
        if package_param_or_def_found and JSON_FILES_BASE["_PACKAGE_FILE_PATH"] == None:
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

    parse_filepaths(data)
    parse_metrics(data)
    config = parse_parameters(data)

    return config


def parse_flow_variables():
    """
    Parse the flow variables from source
    - Code: Makefile `vars` target output

    TODO: Tests.

    Output:
    - flow_variables: set of flow variables
    """

    # first, generate vars.tcl
    cur_path = Path(__file__).resolve().parent
    makefile_path = cur_path / "../../../../flow/"
    initial_path = Path.cwd()
    os.chdir(makefile_path)
    result = subprocess.run(["make", "vars", f"PLATFORM={args.platform}"])
    if result.returncode != 0:
        print(f"[ERROR TUN-0018] Makefile failed with error code {result.returncode}.")
        sys.exit(1)
    if not Path("vars.tcl").is_file():
        print(f"[ERROR TUN-0019] Makefile did not generate vars.tcl.")
        sys.exit(1)
    os.chdir(initial_path)

    # for code parsing, you need to parse from both scripts and vars.tcl file.
    pattern = r"(?:::)?env\((.*?)\)"
    files = glob.glob(str(cur_path / "../../../../flow/scripts/*.tcl"))
    files.append(str(cur_path / "../../../../flow/vars.tcl"))
    variables = set()
    for file in files:
        with open(file) as fp:
            matches = re.findall(pattern, fp.read())
        for match in matches:
            for variable in match.split("\n"):
                variables.add(variable.strip().upper())
    return variables


def parse_config(config, files):
    """
    Parse configuration received from tune into make variables.
    """
    options = ""
    prefix_dicts = {
        "_FR_": dict(), "_SDC_": dict(),
        "_TOP_PARAM_": dict(), "_TOP_DEF_": dict(),
        "_PACKAGE_PARAM_": dict(), "_PACKAGE_DEF_": dict()
    }
    flow_variables = parse_flow_variables()

    if files["_TOP_LEVEL_FILE_PATH"] is not None or files["_PACKAGE_FILE_PATH"] is not None:
        verilog_rewriter = VerilogRewriter(top_fp=files["_TOP_LEVEL_FILE_PATH"], pkg_fp=files["_PACKAGE_FILE_PATH"])

    for key, value in config.items():
        if key.startswith("_"):
            prefix = next((p for p in prefix_dicts if key.startswith(p)), None)
            if prefix:
                prefix_dicts[prefix][key.replace(prefix, "", 1)] = value
            elif key == "_PINS_DISTANCE": # Special substitution
                options += f' PLACE_PINS_ARGS="-min_distance {value}"'
            elif key == "_SYNTH_FLATTEN": # Special substitution
                print("[WARNING TUN-0013] Non-flatten the designs are not fully supported, ignoring _SYNTH_FLATTEN parameter.")
        else:
            # FIXME there is no robust way to get this metainformation from ORFS about the variables, so disable this code for now.

            # Sanity check: ignore all flow variables that are not tunable
            # if key not in flow_variables:
            #     print(f"[ERROR TUN-0017] Variable {key} is not tunable.")
            #     sys.exit(1)
            options += f" {key}={value}"

    if prefix_dicts["_SDC_"]:
        write_sdc(prefix_dicts["_SDC_"], files["_SDC_FILE_PATH"])
        options += f" SDC_FILE={files['_SDC_FILE_PATH']}"
    if prefix_dicts["_FR_"]:
        write_fast_route(prefix_dicts["_FR_"], files["_FR_FILE_PATH"])
        options += f" FASTROUTE_TCL={files['_FR_FILE_PATH']}"
    if prefix_dicts["_TOP_PARAM_"] or prefix_dicts["_TOP_DEF_"] or prefix_dicts["_PACKAGE_PARAM_"] or prefix_dicts["_PACKAGE_DEF_"]:
        verilog_rewriter.update_sv(prefix_dicts["_TOP_DEF_"], prefix_dicts["_TOP_PARAM_"], prefix_dicts["_PACKAGE_DEF_"], prefix_dicts["_PACKAGE_PARAM_"])
    return options

def write_sdc(variables, sdc_file_path):
    """
    Create a SDC file with parameters for current tuning iteration.
    """
    # Handle case where the reference file does not exist
    if not sdc_file_path.is_file():
        print("[ERROR TUN-0020] No SDC reference file provided.")

    with open(sdc_file_path, "r") as file:
        sdc_content = file.read()

    if sdc_content == "":
        print("[ERROR TUN-0020] SDC reference file provided is empty.")
        sys.exit(1)
    for key, value in variables.items():
        if key == "CLK_PERIOD":
            if sdc_content.find("set clk_period") != -1:
                sdc_content = re.sub(
                    r"set clk_period .*\n(.*)", f"set clk_period {value}\n\\1", sdc_content
                )
            else:
                sdc_content = re.sub(
                    r"-period [0-9\.]+ (.*)", f"-period {value} \\1", sdc_content
                )
                sdc_content = re.sub(r"-waveform [{}\s0-9\.]+[\s|\n]", "", sdc_content)
        elif key == "UNCERTAINTY":
            if sdc_content.find("set uncertainty") != -1:
                sdc_content = re.sub(
                    r"set uncertainty .*\n(.*)",
                    f"set uncertainty {value}\n\\1",
                    sdc_content,
                )
            else:
                sdc_content += f"\nset uncertainty {value}\n"
        elif key == "IO_DELAY":
            if sdc_content.find("set io_delay") != -1:
                sdc_content = re.sub(
                    r"set io_delay .*\n(.*)", f"set io_delay {value}\n\\1", sdc_content
                )
            else:
                sdc_content += f"\nset io_delay {value}\n"
    with sdc_file_path.open("w") as file:
        file.write(sdc_content)


def write_fast_route(variables, fr_file_path):
    """
    Create a FastRoute Tcl file with parameters for current tuning iteration.
    """
    # Handle case where the reference file does not exist (asap7 doesn't have reference)
    if not fr_file_path.is_file():
        print("[ERROR TUN-0020] No FastRoute Tcl reference file provided.")
        sys.exit(1)
    with fr_file_path.open("r") as file:
        fr_content = file.read()

    if fr_content == "" and args.platform != "asap7":
        print("[ERROR TUN-0021] FastRoute Tcl reference file provided is empty.")
        sys.exit(1)
    layer_cmd = "set_global_routing_layer_adjustment"
    for key, value in variables.items():
        if key.startswith("LAYER_ADJUST"):
            layer = key.lstrip("LAYER_ADJUST")
            # If there is no suffix (i.e., layer name) apply adjust to all layers.
            if layer == "":
                fr_content += "\nset_global_routing_layer_adjustment"
                fr_content += " $::env(MIN_ROUTING_LAYER)"
                fr_content += "-$::env(MAX_ROUTING_LAYER)"
                fr_content += f" {value}"
            elif re.search(f"{layer_cmd}.*{layer}", fr_content):
                fr_content = re.sub(
                    f"({layer_cmd}.*{layer}).*\n(.*)", f"\\1 {value}\n\\2", fr_content
                )
            else:
                fr_content += f"\n{layer_cmd} {layer} {value}\n"
        elif key == "GR_SEED":
            fr_content += f"\nset_global_routing_random -seed {value}\n"
    with fr_file_path.open("w") as file:
        file.write(fr_content)


def run_command(cmd, timeout=None, stderr_file=None, stdout_file=None, fail_fast=False):
    """
    Wrapper for subprocess.run
    Allows to run shell command, control print and exceptions.
    """
    process = run(
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


@ray.remote
def openroad_distributed(repo_dir, base_log_dir, config):
    """Simple wrapper to run openroad distributed with Ray."""
    copy_dir = base_log_dir / Path(f"/variant-{uuid()}")
    files = copy_repo(repo_dir, copy_dir)
    config = parse_config(config, files)
    os.chdir(copy_dir)
    openroad(copy_dir, config, str(uuid()))


def openroad(base_dir, parameters, flow_variant):
    """
    Run OpenROAD-flow-scripts with a given set of parameters.
    """
    # Make sure path ends in a slash, i.e., is a folder
    flow_variant = f"{args.experiment}/{flow_variant}"
    log_path = os.getcwd() + "/"
    report_path = log_path.replace("logs", "reports")
    run_command(f"mkdir -p {log_path}")
    run_command(f"mkdir -p {report_path}")

    export_command = f"export PATH={BASE_INSTALL_PATH}/OpenROAD/bin"
    export_command += f":{BASE_INSTALL_PATH}/yosys/bin:$PATH"
    export_command += " && "

    make_command = export_command
    make_command += f"make"
    if args.stage_stop != "all":
        for i in range (ALLOWED_OPENROAD_STAGES.index(args.stage_stop) + 1):
            make_command += f" {ALLOWED_OPENROAD_STAGES[i]}" # Append preceding flow stages to make command
    make_command += f" -C {base_dir}/flow DESIGN_CONFIG=designs/"
    make_command += f"{args.platform}/{args.design}/config.mk"
    make_command += f" PLATFORM={args.platform}"
    make_command += f" FLOW_VARIANT={flow_variant} {parameters}"
    make_command += f" EQUIVALENCE_CHECK=0"
    make_command += f" NPROC={args.openroad_threads} SHELL=bash"
    run_command(f"cd {base_dir}")
    run_command(
        make_command,
        timeout=args.timeout,
        stderr_file = Path(log_path) / "error-make-finish.log",
        stdout_file = Path(log_path) / "make-finish-stdout.log"
    )

    metrics_file = os.path.join(report_path, "metrics.json")
    metrics_command = export_command
    metrics_command += f"{base_dir}/flow/util/genMetrics.py -x"
    metrics_command += f" -v {flow_variant}"
    metrics_command += f" -d {args.design}"
    metrics_command += f" -p {args.platform}"
    metrics_command += f" -o {metrics_file}"
    run_command(
        metrics_command,
        stderr_file = Path(log_path) / "error-metrics.log",
        stdout_file = Path(log_path) / "metrics-stdout.log"
    )

    return metrics_file

def run_rtl_sim(base_dir, flow_variant, files):
    """
    Runs the simulation and returns the path to the metrics file.
    """

    flow_variant = f"{args.experiment}/{flow_variant}"
    log_path = os.getcwd() + "/"
    report_path = log_path.replace("logs", "reports")
    sim_report_path = base_dir / "flow" / "reports"
    sim_file_path = files["_SIM_FILE_PATH"]
    
    run_command(f"mkdir -p {log_path}")
    run_command(f"mkdir -p {sim_report_path}")

    rtl_sim_command = ""

    if sim_file_path.suffix == ".py":
        rtl_sim_command += f"python {sim_file_path}"
    elif sim_file_path.suffix == ".sh":
        rtl_sim_command += f"{sim_file_path}"
    elif sim_file_path.name in ["Makefile", "makefile"] or sim_file_path.suffix == ".mk":
        rtl_sim_command += f"make -f {sim_file_path}"
    else:
        print(f"[ERROR TUN-xxxxx] Unsupported simulation file type: {sim_file_path.suffix}")

    run_command(
        rtl_sim_command,
        stderr_file = Path(log_path) / "error-rtl_sim.log",
        stdout_file = Path(sim_report_path) / "rtl_sim-stdout.log"
    )

    metrics_names_file = Path(sim_report_path) / "rtl_sim_metrics_names.json"
    with open(metrics_names_file, "w") as file:
        json.dump(METRICS_CONFIG, file, indent=2)

    metrics_file = sim_report_path / "metrics.json"
    metrics_command = f"{base_dir}/flow/util/genMetrics.py -x"
    metrics_command += f" -v {flow_variant}"
    metrics_command += f" -d {args.design}"
    metrics_command += f" -p {args.platform}"
    metrics_command += f" -o {metrics_file}"
    run_command(
        metrics_command,
        stderr_file = Path(log_path) / "error-metrics.log",
        stdout_file = Path(log_path) / "metrics-stdout.log"
    )

    os.remove(metrics_names_file) # Cleanup metrics name file

    return metrics_file

def clone(path):
    """
    Clone base repo in the remote machine. Only used for Kubernetes at GCP.
    """
    if args.git_clone:
        run_command(f"rm -rf {path}")
    if not os.path.isdir(f"{path}/.git"):
        git_command = "git clone --depth 1 --recursive --single-branch"
        git_command += f" {args.git_clone_args}"
        git_command += f" --branch {args.git_orfs_branch}"
        git_command += f" {args.git_url} {path}"
        run_command(git_command)


def build(base, install):
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
    run_command(build_command)


@ray.remote
def setup_repo(base):
    """
    Clone ORFS repository and compile binaries.
    """
    print(f"[INFO TUN-0000] Remote folder: {base}")
    install = base / "tools/install"
    if args.server is not None:
        clone(base)
    build(base, install)
    return install


def parse_arguments():
    """
    Parse arguments from command line.
    """
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(
        help="mode of execution", dest="mode", required=True
    )
    tune_parser = subparsers.add_parser("tune")
    _ = subparsers.add_parser("sweep")

    # DUT
    parser.add_argument(
        "--design",
        type=str,
        metavar="<gcd,jpeg,ibex,aes,...>",
        required=True,
        help="Name of the design for Autotuning.",
    )
    parser.add_argument(
        "--platform",
        type=str,
        metavar="<sky130hd,sky130hs,asap7,...>",
        required=True,
        help="Name of the platform for Autotuning.",
    )

    # Experiment Setup
    parser.add_argument(
        "--config",
        type=str,
        metavar="<path>",
        required=True,
        help="Configuration file that sets which knobs to use for Autotuning.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        metavar="<str>",
        default="test",
        help="Experiment name. This parameter is used to prefix the"
        " FLOW_VARIANT and to set the Ray log destination.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        metavar="<float>",
        default=None,
        help="Time limit (in hours) for each trial run. Default is no limit.",
    )
    tune_parser.add_argument(
        "--resume", action="store_true", help="Resume previous run."
    )

    # Setup
    parser.add_argument(
        "--git_clean",
        action="store_true",
        help="Clean binaries and build files."
        f"WARNING: may lose previous data."
        " Use carefully.",
    )
    parser.add_argument(
        "--git_clone",
        action="store_true",
        help="Force new git clone."
        f"WARNING: may lose previous data."
        " Use carefully.",
    )
    parser.add_argument(
        "--git_clone_args",
        type=str,
        metavar="<str>",
        default="",
        help="Additional git clone arguments.",
    )
    parser.add_argument(
        "--git_latest", action="store_true", help="Use latest version of OpenROAD app."
    )
    parser.add_argument(
        "--git_or_branch",
        type=str,
        metavar="<str>",
        default="",
        help="OpenROAD app branch to use.",
    )
    parser.add_argument(
        "--git_orfs_branch",
        type=str,
        metavar="<str>",
        default="master",
        help="OpenROAD-flow-scripts branch to use.",
    )
    parser.add_argument(
        "--git_url",
        type=str,
        metavar="<url>",
        default=ORFS_URL,
        help="OpenROAD-flow-scripts repo URL to use.",
    )
    parser.add_argument(
        "--build_args",
        type=str,
        metavar="<str>",
        default="",
        help="Additional arguments given to ./build_openroad.sh.",
    )

    # ML
    tune_parser.add_argument(
        "--algorithm",
        type=str,
        choices=["hyperopt", "ax", "optuna", "pbt", "random"],
        default="hyperopt",
        help="Search algorithm to use for Autotuning.",
    )
    tune_parser.add_argument(
        "--eval",
        type=str,
        choices=["default", "score-improv"],
        default="default",
        help="Evaluate function to use with search algorithm.",
    )
    tune_parser.add_argument(
        "--samples",
        type=int,
        metavar="<int>",
        default=10,
        help="Number of samples for tuning.",
    )
    tune_parser.add_argument(
        "--iterations",
        type=int,
        metavar="<int>",
        default=1,
        help="Number of iterations for tuning.",
    )
    tune_parser.add_argument(
        "--resources_per_trial",
        type=int,
        metavar="<int>",
        default=1,
        help="Number of CPUs to request for each tuning job.",
    )
    tune_parser.add_argument(
        "--reference",
        type=str,
        metavar="<path>",
        default=None,
        help="Reference file for use with ScoreImprov.",
    )
    tune_parser.add_argument(
        "--perturbation",
        type=int,
        metavar="<int>",
        default=25,
        help="Perturbation interval for PopulationBasedTraining.",
    )
    tune_parser.add_argument(
        "--seed",
        type=int,
        metavar="<int>",
        default=42,
        help="Random seed. (0 means no seed.)",
    )

    # Workload
    parser.add_argument(
        "--jobs",
        type=int,
        metavar="<int>",
        default=int(np.floor(cpu_count() / 2)),
        help="Max number of concurrent jobs.",
    )
    parser.add_argument(
        "--openroad_threads",
        type=int,
        metavar="<int>",
        default=16,
        help="Max number of threads openroad can use.",
    )
    parser.add_argument(
        "--server",
        type=str,
        metavar="<ip|servername>",
        default=None,
        help="The address of Ray server to connect.",
    )
    parser.add_argument(
        "--port",
        type=int,
        metavar="<int>",
        default=10001,
        help="The port of Ray server to connect.",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Verbosity level.\n\t0: only print Ray status\n\t1: also print"
        " training stderr\n\t2: also print training stdout.",
    )

    parser.add_argument(
        "--stage_stop",
        type=str,
        metavar="<all, floorplan, place, cts, route, rtl_sim...>",
        default="all",
        help="Stage at which to stop the flow and use for metrics.",
        choices=ALLOWED_OPENROAD_STAGES + ALLOWED_SIM_STAGES
    )

    arguments = parser.parse_args()
    if arguments.mode == "tune":
        arguments.algorithm = arguments.algorithm.lower()
        # Validation of arguments
        if arguments.eval == "score-improv" and arguments.reference is None:
            print(
                '[ERROR TUN-0006] The argument "--eval score-improv"'
                ' requires that "--reference <FILE>" is also given.'
            )
            sys.exit(7)

    arguments.experiment += f"-{arguments.mode}-{DATE}"

    if arguments.timeout is not None:
        arguments.timeout = round(arguments.timeout * 3600)

    if (arguments.mode == "tune") and (arguments.eval == "default") and (arguments.stage_stop in ["floorplan", "place", "cts"]):
        print(f"Score for evaluation method 'default' with stage stop `{arguments.stage_stop}` will not consider DRC errors (only available after routing).")

    return arguments


def set_algorithm(experiment_name, config):
    """
    Configure search algorithm.
    """
    # Pre-set seed if user sets seed to 0
    if args.seed == 0:
        print(
            f"Warning: you have chosen not to set a seed. Do you wish to continue? (y/n)"
        )
        if input().lower() != "y":
            sys.exit(0)
        args.seed = None
    else:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    if args.algorithm == "hyperopt":
        algorithm = HyperOptSearch(
            points_to_evaluate=best_params,
            random_state_seed=args.seed,
        )
    elif args.algorithm == "ax":
        ax_client = AxClient(
            enforce_sequential_optimization=False,
            random_seed=args.seed,
        )
        AxClientMetric = namedtuple("AxClientMetric", "minimize")
        ax_client.create_experiment(
            name=experiment_name,
            parameters=config,
            objectives={METRIC: AxClientMetric(minimize=True)},
        )
        algorithm = AxSearch(ax_client=ax_client, points_to_evaluate=best_params)
    elif args.algorithm == "optuna":
        algorithm = OptunaSearch(points_to_evaluate=best_params, seed=args.seed)
    elif args.algorithm == "pbt":
        print("Warning: PBT does not support seed values. args.seed will be ignored.")
        algorithm = PopulationBasedTraining(
            time_attr="training_iteration",
            perturbation_interval=args.perturbation,
            hyperparam_mutations=config,
            synch=True,
        )
    elif args.algorithm == "random":
        algorithm = BasicVariantGenerator(
            max_concurrent=args.jobs,
            random_state=args.seed,
        )

    # A wrapper algorithm for limiting the number of concurrent trials.
    if args.algorithm not in ["random", "pbt"]:
        algorithm = ConcurrencyLimiter(algorithm, max_concurrent=args.jobs)

    return algorithm


def set_best_params(platform, design):
    """
    Get current known best parameters if it exists.
    """
    params = []
    best_param_file = Path("designs") / platform / design / "autotuner-best.json"
    if os.path.isfile(best_param_file):
        with open(best_param_file) as file:
            params = json.load(file)
    return params


def set_training_class(function):
    """
    Set training class.
    """
    if function == "default":
        return AutoTunerBase
    if function == "score-improv":
        return ScoreImprov
    return None


@ray.remote
def save_best(results):
    """
    Save best configuration of parameters found.
    """

    best_config = results.best_config
    best_config["score_metrics_config"] = METRICS_CONFIG
    best_config["best_result"] = results.best_result[METRIC]
    trial_id = results.best_trial.trial_id
    new_best_path = BASE_LOCAL_DIR / args.experiment / f"autotuner-best-{trial_id}.json"
    with open(new_best_path, "w") as new_best_file:
        json.dump(best_config, new_best_file, indent=4)
    print(f"[INFO TUN-0003] Best parameters written to {new_best_path}")


@ray.remote
def consumer(queue):
    """consumer"""
    while not queue.empty():
        next_item = queue.get()
        params = next_item[2]
        print(f"[INFO TUN-0007] Scheduling run for parameters {params}.")
        ray.get(openroad_distributed.remote(*next_item))
        print(f"[INFO TUN-0008] Finished run for parameters {params}.")


def sweep():
    """Run sweep of parameters"""
    if args.server is not None:
        # For remote sweep we create the following directory structure:
        #      1/     2/         3/       4/
        # <repo>/<logs>/<platform>/<design>/
        repo_dir = (BASE_LOCAL_DIR / "../" * 4).resolve()
    else:
        repo_dir = Path("../").resolve()
        log_dir = BASE_LOCAL_DIR / f"sweep-{DATE}"

    print(f"[INFO TUN-0012] Log folder {log_dir}.")
    os.makedirs(log_dir, exist_ok=True)

    queue = Queue()
    parameter_list = list()

    for name, content in config_dict.items():
        if content["type"] == "choice":
            parameter_list.append([{name: i} for i in content["values"]])
        else:
            if content["step"] == 0:
                print(f"[ERROR TUN-0014] Sweep does not support step value zero.")
                sys.exit(1)
            if (content["minmax"][0] == content["minmax"][1]):
                parameter_list.append([{name: content["minmax"][0]}])
            else:
                parameter_list.append([{name: i} for i in np.arange(start=content["minmax"][0],
                                                                    stop=content["minmax"][1] + content["step"],
                                                                    step=content["step"])])

    parameter_list = list(product(*parameter_list))

    for parameter in parameter_list:
        run_params = dict()
        for value in parameter:
            run_params.update(value)
        queue.put([repo_dir, log_dir, run_params])

    workers = [consumer.remote(queue) for _ in range(args.jobs)]
    print("[INFO TUN-0009] Waiting for results.")
    ray.get(workers)
    print("[INFO TUN-0010] Sweep complete.")


if __name__ == "__main__":
    args = parse_arguments()

    # Read config and original files before handling where to run in case we
    # need to upload the files.
    config_dict = read_config(os.path.abspath(args.config))

    # Connect to remote Ray server if any, otherwise will run locally
    # TODO: Fix directory structure for server runs.
    if args.server is not None:
        # At GCP we have a NFS folder that is present for all worker nodes.
        # This allows to build required binaries once. We clone, build and
        # store intermediate files at BASE_LOCAL_DIR.
        with open(args.config) as config_file:
            if args.git_or_branch != "":
                BASE_LOCAL_DIR = f"/shared-data/autotuner-orfs-{args.git_orfs_branch}-or-{args.git_or_branch}"
            if args.git_latest:
                BASE_LOCAL_DIR = f"/shared-data/autotuner-orfs-{args.git_orfs_branch}-or-latest"
        # Connect to ray server before first remote execution.
        ray.init(f"ray://{args.server}:{args.port}")
        # Remote functions return a task id and are non-blocking. Since we
        # need the setup repo before continuing, we call ray.get() to wait
        # for its completion.
        BASE_INSTALL_PATH = ray.get(setup_repo.remote(BASE_LOCAL_DIR))
        BASE_LOCAL_DIR = BASE_LOCAL_DIR / "/flow/logs/{args.platform}/{args.design}"
        print("[INFO TUN-0001] NFS setup completed.")
    else:
        # For local runs, use the same folder as other ORFS utilities.
        BASE_ORFS_FLOW_DIR = Path(__file__).resolve().parents[4] / "flow"
        os.chdir(BASE_ORFS_FLOW_DIR)
        BASE_LOCAL_DIR = BASE_ORFS_FLOW_DIR / "logs" / args.platform / args.design
        BASE_INSTALL_PATH = Path("/foss/tools")

    if args.mode == "tune":
        best_params = set_best_params(args.platform, args.design)
        search_algo = set_algorithm(args.experiment, config_dict)
        TrainClass = set_training_class(args.eval)
        # ScoreImprov requires a reference file to compute training scores.
        if args.eval == "score-improv":
            reference = ScoreImprov.read_metrics(args.reference)

        tune_args = dict(
            name=args.experiment,
            metric=METRIC,
            mode="min",
            num_samples=args.samples,
            fail_fast=False,
            storage_path=str(BASE_LOCAL_DIR),
            resume=args.resume,
            stop={"training_iteration": args.iterations},
            resources_per_trial={"cpu": args.resources_per_trial},
            log_to_file=["trail-out.log", "trail-err.log"],
            trial_name_creator=lambda x: f"variant-{x.trainable_name}-{x.trial_id}-ray",
            trial_dirname_creator=lambda x: f"variant-{x.trainable_name}-{x.trial_id}-ray",
        )
        if args.algorithm == "pbt":
            os.environ["TUNE_MAX_PENDING_TRIALS_PG"] = str(args.jobs)
            tune_args["scheduler"] = search_algo
        else:
            tune_args["search_alg"] = search_algo
            tune_args["scheduler"] = AsyncHyperBandScheduler()
        if args.algorithm != "ax":
            tune_args["config"] = config_dict

        analysis = tune.run(TrainClass, **tune_args)            

        task_id = save_best.remote(analysis)
        _ = ray.get(task_id)
        print(f"[INFO TUN-0002] Best parameters found: {analysis.best_config}")

        # if all runs have failed
        if analysis.best_result["minimum"] == ERROR_METRIC:
            print(f"[ERROR TUN-0016] No successful runs found.")
            sys.exit(1)
    elif args.mode == "sweep":
        sweep()
