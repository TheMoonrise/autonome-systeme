import requests
import json
import os
import numpy as np
from src.plots.plots import Plots
from argparse import ArgumentParser

parser = ArgumentParser(description='The plotting arg parser.')
parser.add_argument('--run_id', type=str, nargs='+', help='run id of the run you want to plot')
parser.add_argument('--key', type=str, help='metric you want to plot')
args = parser.parse_args()

run_id = args.run_id
key = args.key

values = []


def run_values(data):
    step = 1
    values = []
    step_values = []

    for i, x in enumerate(data):
        while int(x['step']) > step:
            if not step_values:
                if step == 1: values.append(0)
                else: values.append(values[step - 2])
            else:
                values.append(np.mean(step_values))
                step_values.clear()
            step += 1
        step_values.append(x['value'])
    return values


for run in run_id:
    file_name = os.path.join(os.path.dirname(__file__), f'../metrics/metrics_{run}_{key}.json')
    os.makedirs(os.path.join(os.path.dirname(file_name)), exist_ok=True)
    if os.path.isfile(file_name):
        # read file if it already exists
        with open(file_name, 'r') as f:
            data = json.load(f)
    else:
        # get data from mlflow api
        assert None not in (run, key)
        url = f'http://157.245.22.106:5000/api/2.0/mlflow/metrics/get-history?run_id={run}&metric_key={key}'
        data = requests.get(url).json()['metrics']
        with open(file_name, 'w') as f:
            json.dump(data, f)

    # values.append([x['value'] for x in data])
    values.append(run_values(data))

for x in values: print(len(x))
s_dir = os.path.join(os.path.dirname(__file__), '../plots')
os.makedirs(s_dir, exist_ok=True)
plots = Plots(s_dir, 'eval')
plots.plot_performance(values, title=f'{key}_{run}')
for value in values:
    plots.plot_moving_avg_performance(value, title=f'{key}_{run}')
    plots.plot_moving_avg_std_performance(value, title=f'{key}_{run}')
