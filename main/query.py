import requests
import json
import os
from src.plots.plots import Plots
from argparse import ArgumentParser

parser = ArgumentParser(description='The plotting arg parser.')
parser.add_argument('--run_id', type=str, help='run id of the run you want to plot')
parser.add_argument('--key', type=str, help='metric you want to plot')
args = parser.parse_args()

run_id = args.run_id
key = args.key
file_name = f'metrics_{run_id}_{key}.json'

if os.path.isfile(file_name):
    # read file if it already exists
    with open(f'metrics_{run_id}_{key}.json', 'r') as f:
        data = json.load(f)
else:
    # get data from mlflow api
    url = f'http://157.245.22.106:5000/api/2.0/mlflow/metrics/get-history?run_id={run_id}&metric_key={key}'
    data = requests.get(url).json()['metrics']
    with open(file_name, 'w') as f:
        json.dump(data, f)

values = [x['value'] for x in data]

s_dir = os.path.join(os.getcwd(), 'plots')
os.makedirs(s_dir, exist_ok=True)
plots = Plots(s_dir, 'eval')
plots.plot_moving_avg_performance(values, title=f'{key}_{run_id}')
