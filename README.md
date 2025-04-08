# Setup

## Python environment
Create a new Python environment and install dependencies. For example, you could use `conda` and `pip`: 
```bash
conda create -n min-delay-charging python=3.11
conda activate min-delay-charging
conda install pip
pip install -r requirements.txt
```

Then, install the `min_delay_charging` package from the root directory:

`pip install .`

## API keys
Running this code requires the use of some personal API keys. These are handled as secrets stored in a `.env` file.


