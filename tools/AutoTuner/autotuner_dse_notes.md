## Autotuner for Design Space Exploration
Goal: Allow automatic exploration of RTL-side parametrization. Particularly useful for open-source distributed IP and determining how best to integrate it in your design.
Also, can be used as as a "regression evaluation" tool that runs periodically on a given codebase and lets the designers ensure the design's PPA is within expectations.

### In progress
- __SV rewrite__: At every iteration, rewrite port parameters and `defines in the top-level SV file.
    - Only rewrites highest-level SV for now. This is mainly to allow for reasonably organized parameters as there are often constraints to respect
    (i.e. output width >= input width, etc.). We considered enforcing constraints programatically by having the user implemented a Python class in AT
    that defines how to iterate on the parameters, but decided against it to limit the amount of Python the hardware designer would have to write. Open to discuss.
    - Port parameters are defined in the config .json like existing parameters, prepended by `_PARAM` or `_DEF_`. Example:
        ```
            "_PARAM_coef_width": {
            "type": "int",
            "minmax": [
                2,
                16
            ],
            "step": 1
            },
        ```
    - Look at `src/autotuner/Rewriter.py` and `src/autotuner/distributed:482` in `sv_rewriter` branch.
    - We find the location of the values of interest and replace them in the SV file (saving a backup of the original file of course).
- __Allow to select which stage to stop at__: New argument to tell Autotuner at which stage in the flow to stop and use results from.
    -Time-saving measure. We consider a typical use-case of DSE'ed Autotuner to be still rather early in the design process which time is most precious.
    -Support `[floorplan, place, cts, route, all]`
    -Look at `select_stage_stop` branch.

### To be implemented/discussed
- __Gather performance metrics for testbench__: I'm thinking of augmenting AT to run benchmarks/testbench to gather more meaningful metrics from that.
    - For the purpose of design space exploration, I think it's important to be able to assess performance on a specific set of benchmarks rather than just using slack as a proxy for performance because meaningful performance comes from both architecture and physical timing characteristics.
    - For this, I'd like to integrate it with at least CocoTB. I know CocoTB isn't nearly robust enough as a verification tool, but the goal for DSE isn't to run 100% exhaustive and formally complete benchmarks but rather to evaluate performance on applications, and I think CocoTB is most appropriate for that.
    - We can couple benchmark completion time with # of gates from synthesis and/or area and from floorplanning to get PPA figures meaningful enough to steer the design towards an optimal solution.
    - I myself have examples of an edge transformer accelerator where I would have benefited from evaluating the performance and area tradeoffs from CocoTB testbenches.
    - We could also run SV testbenches for better compatibility with existing codebases.
    - We could define a standard interface for reporting/ingesting richer metrics from these benchmarks (# of cache misses, # of stalls, arithmetic accuracy, etc.)
- __More insightful reporting__: For design-side parametrization, I think determining what is the "optimal" solution is more "subjective" so we should probably consider exporting things like graphs of metrics vs. parameter in addition to the current set of ideal values that AT exports.
- __Support more metrics from flow__:
    - Things like deepest logic path
    - Upper/lower bounds on area