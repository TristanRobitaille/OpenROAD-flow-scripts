# Instructions for AutoTuner with Ray

_AutoTuner_ is a "no-human-in-loop" parameter tuning framework for commercial and academic RTL-to-GDS flow.
AutoTuner provides a generic interface where users can define parameter configuration as JSON objects.
This enables AutoTuner to easily support various tools and flows. AutoTuner also utilizes [METRICS2.1](https://github.com/ieee-ceda-datc/datc-rdf-Metrics4ML) to capture metrics
of individual search trials. With the abundant features of METRICS2.1, users can explore various reward functions that steer the flow autotuning to different goals.
AutoTuner can source the user-defined metrics from the RTL-to-GDS flow (stopping at a select stage) or from an RTL simulation.

AutoTuner provides the following main functionalities.
* Automatic RTL-to-GDS flow hyperparameter tuning framework based on the results of OpenROAD-flow-script (ORFS) or RTL simulation
* Automatic parameter and define tuning framework for Verilog design (top-level and package) for Design Space Exploration (DSE)
* Parametric sweeping experiments for ORFS

AutoTuner contains top-level Python script for ORFS, each of which implements a different search algorithm. Current supported search algorithms are as follows.
* Random/Grid Search
* Population Based Training ([PBT](https://www.deepmind.com/blog/population-based-training-of-neural-networks))
* Tree Parzen Estimator ([HyperOpt](https://hyperopt.github.io/hyperopt))
* Bayesian + Multi-Armed Bandit ([AxSearch](https://ax.dev/))
* Tree Parzen Estimator + Covariance Matrix Adaptation Evolution Strategy ([Optuna](https://optuna.org/))
* Evolutionary Algorithm ([Nevergrad](https://github.com/facebookresearch/nevergrad))

User-defined metrics and coefficient values are passed-in in the configuration `.json` file. The user can define their own score function in the `AutoTunerBase.get_score()` function.
By default, the score is based on the PPA of the design.

AutoTuner uses Slang to overwrite `parameter` and `define` in a specified Verilog/SystemVerilog (usually the top-level or a wrapper) file and in a specified Verilog/SystemVerilog package file.
This lets the user evaluate different design configurations.


## Setting up AutoTuner

We have provided two convenience scripts, `./installer.sh` and `./setup.sh`
that works in Python3.8 for installation and configuration of AutoTuner,
as shown below:

```{note}
Make sure you run the following commands in the ORFS root directory.
```

```shell
# Install prerequisites
./tools/AutoTuner/installer.sh

# Start virtual environment
./tools/AutoTuner/setup.sh
```

## Input JSON structure

Sample JSON [file](https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts/blob/master/flow/designs/sky130hd/jpeg/autotuner.json) for Sky130HD `jpeg` design:

Alternatively, here is a minimal example to get started:

```json
{
    "parameters" : {
        "_TOP_PARAM_coef_width": {
            "type": "int",
            "minmax": [
                2,
                16
            ],
            "step": 1
        },
        "_PACKAGE_PARAM_ADDR_WIDTH": {
            "type": "int",
            "minmax": [
                8,
                16
            ],
            "step": 2
        },
        "_PACKAGE_DEF_CLOCK_FREQUENCY": {
            "type": "int",
            "minmax": [
                500000000,
                1000000000
            ],
            "step": 100000000
        },
        "_PACKAGE_PARAM_DEFAULT_STATE": {
            "type": "choice",
            "values": [
                "IDLE",
                "INIT",
                "ACTIVE",
                "ERROR",
                "SHUTDOWN"
            ]
        },
        "_SDC_CLK_PERIOD": {
            "type": "float",
            "minmax": [
                7.0,
                9.0
            ],
            "step": 0
        },
        "CORE_UTILIZATION": {
            "type": "int",
            "minmax": [
                20,
                50
            ],
            "step": 1
        }
    },
    "files" : {
        "_SDC_FILE_PATH": "constraint.sdc",
        "_FR_FILE_PATH": "fastroute.tcl",
        "_TOP_LEVEL_FILE_PATH": "../../src/jpeg/jpeg_encoder.v",
        "_PACKAGE_FILE_PATH": "../../src/jpeg/jpeg_pkg.sv",
        "_SIM_FILE_PATH": ""
    },
    "score_metrics_config" : {
        "performance" : {
            "coeff": 10.5
        },
        "power" : {
            "coeff": 2
        },
        "branch_pred_miss_rate" : {
            "coeff": 1
        }
    }
}
```

* `"_TOP_PARAM_coef_width"`, `"_PACKAGE_PARAM_ADDR_WIDTH"`, `"_PACKAGE_DEF_CLOCK_FREQUENCY"`, `"_PACKAGE_PARAM_DEFAULT_STATE"`, `"_SDC_CLK_PERIOD"`, `"CORE_UTILIZATION"`: Parameter names for sweeping/tuning.
* `"type"`: Parameter type ("float", "int", "power_of_2", "choice") for sweeping/tuning. Note that "power_of_2" requires both the min and max to be powers of 2 and will only generate ints.
* `"minmax"`: Min-to-max range for sweeping/tuning. The unit follows the default value of each technology std cell library (only needed for types "float" or "int").
* `"step"`: Parameter step within the minmax range. Step 0 for type "float" means continuous step for sweeping/tuning (only needed for types "float" or "int"). Step 0 for type "int" means the constant parameter.
* `"values"`: List of choices (only needed for type `"choice"`).

## Tunable / sweepable parameters (for ORFS)

Tables of parameters that can be swept/tuned in technology platforms supported by ORFS.
Any variable that can be set from the command line can be used for tune or sweep.

For SDC you can use:

* `_SDC_FILE_PATH`
  - Path relative to the current JSON file to the SDC file.
* `_SDC_CLK_PERIOD`
  - Design clock period. This will create a copy of `_SDC_FILE_PATH` and modify the clock period.
* `_SDC_UNCERTAINTY`
  - Clock uncertainty. This will create a copy of `_SDC_FILE_PATH` and modify the clock uncertainty.
* `_SDC_IO_DELAY`
  - I/O delay. This will create a copy of `_SDC_FILE_PATH` and modify the I/O delay.


For Global Routing parameters that are set on `fastroute.tcl` you can use:

* `_FR_FILE_PATH`
  - Path relative to the current JSON file to the `fastroute.tcl` file.
* `_FR_LAYER_ADJUST`
  - Layer adjustment. This will create a copy of `_FR_FILE_PATH` and modify the layer adjustment for all routable layers, i.e., from `$MIN_ROUTING_LAYER` to `$MAX_ROUTING_LAYER`.
* `_FR_LAYER_ADJUST_NAME`
  - Layer adjustment for layer NAME. This will create a copy of `_FR_FILE_PATH` and modify the layer adjustment only for the layer NAME.
* `_FR_GR_SEED`
  - Global route random seed. This will create a copy of `_FR_FILE_PATH` and modify the global route random seed.

## Defining design parameters and defines

The same configuration file and item format as that for specifying the ORFS parameters is used to specifying the `parameter` and `define` to tune. For the top-level file, prepend the name with `_TOP_PARAM_` and `_TOP_DEF` for parameters and defines, respectively. For the package file, prepend the name with `_PACKAGE_PARAM_` and `_PACKAGE_DEF` for parameters and defines, respectively.

The top-level file to use is specified in the `.json` file with the field  `"_TOP_LEVEL_FILE_PATH"` and the package file is defined specified with the field `"_PACKAGE_FILE_PATH"`. Note that the path must be relative to the location of the `.json`.

## Defining metrics used to compute the score of a run

The user can define their metrics used to compute the score of a run. The metrics are defined in the configuration `.json` and the function to compute the score is defined in the `ScoreImprov.get_score()` method. By default, the score is the PPA and is computed using the coefficients shown in the above JSON snippet. The metrics come from the output of the selected OpenRoad flow stage or the RTL simulation.

To run an RTL simulation in addition to the OpenRoad flow, call AutoTuner with the `run_sim` argument. For this, a file path to an executable simulation (`.py`, `.sh` or a Makefile) is required in the `.json` file (under `"_SIM_FILE_PATH"`). If using a `.sh` script, make sure to have the correct execute permissions.
The simulation is required to output metrics as defined in the `score_metrics_config` group of the `.json` configuration file. The script parse the stdout output of the simulation, which must contain, for example:
```shell
branch_pred_miss_rate: 0.315149
percent_cache_miss: 0.11828
```

You can also avoid running the OpenROAD flow by setting `stage_stop` to `"none"`.

## How to use

### General Information

The `distributed.py` script located in `./tools/AutoTuner/src/autotuner` uses [Ray's](https://docs.ray.io/en/latest/index.html) job scheduling and management to
fully utilize available hardware resources from a single server
configuration, on-premise or over the cloud with multiple CPUs.

The two modes of operation:
- `sweep`, where every possible parameter combination in the search space is tested
- `tune`, where we use Ray's Tune feature to intelligently search the space and optimize hyperparameters using one of the algorithms listed above.

The `sweep` mode is useful when we want to isolate or test a single or very few
parameters. On the other hand, `tune` is more suitable for finding
the best combination of a complex and large number of flow
parameters. Both modes rely on user-specified search space that is
defined by a `.json` file, they use the same syntax and format,
though some features may not be available for sweeping.

#### Notes
```{note}
* The order of the parameters matter. Arguments `--design`, `--platform` and `--config` are always required and should precede *mode*.
* The design files must be located in the `OpenRoad-flow-scripts` repo directory. Typically, the configuration files and `.tcl` scripts are located in `flow/designs/<platform>/<design>` and the source files are located in `flow/designs/src/<design>`.
```

#### Tune only

* AutoTuner: `python3 distributed.py tune -h`

Example:

```shell
python3 distributed.py --design gcd --platform sky130hd \
                       --config ../../../../flow/designs/sky130hd/gcd/autotuner.json \
                       tune --samples 5
```
#### Sweep only

* Parameter sweeping: `python3 distributed.py sweep -h`

Example:

```shell
python3 distributed.py --design gcd --platform sky130hd \
                       --config distributed-sweep-example.json \
                       sweep
```


### Google Cloud Platform (GCP) distribution with Ray

GCP Setup Tutorial coming soon.


### List of input arguments
| Argument                      | Description                                                                                                   | Default |
|-------------------------------|---------------------------------------------------------------------------------------------------------------|---------|
| `--design`                    | Name of the design for Autotuning.                                                                            ||
| `--platform`                  | Name of the platform for Autotuning.                                                                          ||
| `--config`                    | Configuration file that sets which knobs to use for Autotuning.                                               ||
| `--stage_stop`                | Flow stage at which to stop script. May be one of `["floorplan", "place", "cts", "route", "all", "none"]`     | all |
| `--run_sim`                   | Additionally run an arbitrary (RTL, functional, etc.) simulation to collect additional metrics                | false |
| `--experiment`                | Experiment name. This parameter is used to prefix the FLOW_VARIANT and to set the Ray log destination.        | test |
| `--git_clean`                 | Clean binaries and build files. **WARNING**: may lose previous data.                                          ||
| `--git_clone`                 | Force new git clone. **WARNING**: may lose previous data.                                                     ||
| `--git_clone_args`            | Additional git clone arguments.                                                                               ||
| `--git_latest`                | Use latest version of OpenROAD app.                                                                           ||
| `--git_or_branch`             | OpenROAD app branch to use.                                                                                   ||
| `--git_orfs_branch`           | OpenROAD-flow-scripts branch to use.                                                                          ||
| `--git_url`                   | OpenROAD-flow-scripts repo URL to use.                                                                        | [ORFS GitHub repo](https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts) |
| `--build_args`                | Additional arguments given to ./build_openroad.sh                                                             ||
| `--samples`                   | Number of samples for tuning.                                                                                 | 10 |
| `--jobs`                      | Max number of concurrent jobs.                                                                                | # of CPUs / 2 |
| `--openroad_threads`          | Max number of threads usable.                                                                                 | 16 |
| `--server`                    | The address of Ray server to connect.                                                                         ||
| `--port`                      | The port of Ray server to connect.                                                                            | 10001 |
| `--timeout`                   | Time limit (in hours) for each trial run.                                                                     | No limit |
| `-v` or `--verbose`           | Verbosity Level. [0: Only ray status, 1: print stderr, 2: print stdout on top of what is in level 0 and 1. ]  | 0 |
|                               |                                                                                                               ||

#### Input arguments specific to tune mode
The following input arguments are applicable for tune mode only.

| Argument                      | Description                                                                                                   | Default |
|-------------------------------|---------------------------------------------------------------------------------------------------------------|---------|
| `--algorithm`                 | Search algorithm to use for Autotuning.                                                                       | hyperopt |
| `--eval`                      | Evaluate function to use with search algorithm.                                                               ||
| `--iterations`                | Number of iterations for tuning.                                                                              | 1 |
| `--resources_per_trial`       | Number of CPUs to request for each tuning job.                                                                | 1 |
| `--reference`                 | Reference file for use with ScoreImprov.                                                                      ||
| `--perturbation`              | Perturbation interval for PopulationBasedTraining                                                             | 25 |
| `--seed`                      | Random seed.                                                                                                  | 42 |
| `--resume`                    | Resume previous run.                                                                                          ||
|                               |                                                                                                               ||

### GUI

Basically, progress is displayed at the terminal where you run, and when all runs are finished, the results are displayed.
You could find the "Best config found" on the screen.

To use TensorBoard GUI, run `tensorboard --logdir=./<logpath>`. While TensorBoard is running, you can open the webpage `http://localhost:6006/` to see the GUI.

We show three different views possible at the end, namely: `Table View`, `Scatter Plot Matrix View` and `Parallel Coordinate View`.

![Table View](../images/Autotuner_Table_view.webp)
<p style="text-align: center;">Table View</p>

![Scatter Plot Matrix View](../images/Autotuner_scatter_plot_matrix_view.webp)
<p style="text-align: center;">Scatter Plot Matrix View</p>

![Parallel Coordinate View](../images/Autotuner_best_parameter_view.webp)
<p style="text-align: center;">Parallel Coordinate View (best run is in green)</p>

## Testing framework

Assuming the virtual environment is setup at `./tools/AutoTuner/autotuner_env`:

```
./tools/AutoTuner/setup.sh
python3 ./tools/AutoTuner/test/smoke_test_sweep.py
python3 ./tools/AutoTuner/test/smoke_test_tune.py
python3 ./tools/AutoTuner/test/smoke_test_sample_iteration.py
```

## Citation

Please cite the following paper.

* J. Jung, A. B. Kahng, S. Kim and R. Varadarajan, "METRICS2.1 and Flow Tuning in the IEEE CEDA Robust Design Flow and OpenROAD", [(.pdf)](https://vlsicad.ucsd.edu/Publications/Conferences/388/c388.pdf), [(.pptx)](https://vlsicad.ucsd.edu/Publications/Conferences/388/c388.pptx), [(.mp4)](https://vlsicad.ucsd.edu/Publications/Conferences/388/c388.mp4), Proc. ACM/IEEE International Conference on Computer-Aided Design, 2021.

## Acknowledgments

AutoTuner has been developed by UCSD with the OpenROAD Project. Additional features by the Integrated Systems Laboratory (IIS) at ETH Zürich.
