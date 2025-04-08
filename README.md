# min-delay-charging
This repository contains Python code to optimize battery-electric bus opportunity charging schedules as described in the *Communications in Transport Research* manuscript "Minimum-delay opportunity charging scheduling for electric buses" by Dan McCabe, Xuegang (Jeff) Ban, and Balázs Kulcsár.

Scripts to replicate the case studies in the paper are under [scripts/simple_case_study.py](scripts/simple_case_study.py) and [scripts/south_king_county.py](scripts/south_king_county.py). Source code for the charging scheduling algorithms and data management are in the [min_delay_charging](min_delay_charging) directory.

## Dependencies
This code base uses `gurobipy` for mixed-integer and linear programming. As such, a Gurobi license must be configured on your machine to run it.

We also rely on some free services that require registration for an API key: OpenRouteService for estimating distances on a road network, and Mapbox for producing interactive plots on maps using `plotly`. 

Because these APIs have limited usage allowed, users of this repo need to provide their own API keys. These API keys should be stored in a `.env` file in the root directory of the repo (i.e., `min-delay-charging/.env`). We use the `python-dotenv` package to read these in as environment variables where needed.

The `.env` file should be formatted as follows with the following entries:

```
# Openrouteservice
ORS_KEY=your_ors_key
# Mapbox (used via Plotly)
MAPBOX_KEY=your_mapbox_key
```

To obtain these keys, follow the steps for each provider. See https://openrouteservice.org/dev/#/signup for Openrouteservice and https://docs.mapbox.com/help/getting-started/access-tokens/ for Mapbox.


## Python environment
Once you've obtained the necessary licenses and API keys, create a new Python environment and install dependencies. For example, you could use `conda` and `pip`: 
```commandline
conda create -n min-delay-charging python=3.11
conda activate min-delay-charging
conda install pip
pip install -r requirements.txt
```

Then, install the `min_delay_charging` package from the root directory:

`pip install .`

This step is important to ensure you can import functions and classes from `min_delay_charging`. It will also automatically unzip the GTFS data included with the repository into its expected location so that you can run the King County case study with the proper input data.

## Running Case Studies
With your Python environment configured as above, the case study scripts should run smoothly. You can run them from the command line:

### Simple Case
```commandline
python scripts/simple_case_study.py
```
The script will print results including objective values and solution times for each of the three methods considered (direct solution with Gurobi, 3S heuristic, and combinatorial Benders decomposition).

### South King County Case
```commandline
python scripts/south_king_county.py
```
In addition to printing results, this script will also use `plotly` to generate figures showing the transit system under study and charger utilization in Scenarios A and B.


