import requests
import json
import os
import numpy as np
from src.plots.plots import Plots
from argparse import ArgumentParser

parser = ArgumentParser(description='The plotting arg parser.')
parser.add_argument('--run_id', type=str, nargs='+', help='run id of the run you want to plot')
parser.add_argument('--key', type=str, help='metric you want to plot')
parser.add_argument('--concat', action="store_true")
parser.add_argument('--params', type=str)


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


params = []
for run in run_id:

    if args.params:
        params_url = f'http://157.245.22.106:5000/api/2.0/mlflow/runs/get?run_id={run}'
        params_data = requests.get(params_url).json()['run']['data']['params']
        params.append(float(next(x['value'] for x in params_data if x['key'] == args.params)))

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
    if args.concat:
        values += run_values(data)
    else:
        values.append(run_values(data))

s_dir = os.path.join(os.path.dirname(__file__), '../plots')
os.makedirs(s_dir, exist_ok=True)
plots = Plots(s_dir, 'eval')
plots.plot_performance(values, title=f'{key}_{run}')


if args.concat:
    plots.plot_moving_avg_performance(values, title=f'{key}_{run}')
    plots.plot_moving_avg_std_performance(values, title=f'{key}_{run}')

else:
    for value in values:
        plots.plot_moving_avg_performance(value, title=f'{key}_{run}')
        plots.plot_moving_avg_std_performance(value, title=f'{key}_{run}')
        pass

plots.plot_performance_curves(values=values, params=params, title=f'{key}_{args.params}')


# Nancy origina


#e330eeac7e7244b4b2bdd2cb1e9c30cb cca29069662141e8881baa0639e85b03 dc4d6525687f4d948fd46e5512ab887d c8a3a7b3736d44a4a23c68929117919c

# 46e1e71758164cd9b23ca2da49c4c42a e574388998144dabbb220aecf1fc2fde d0ea63e424424df8a30f003761ab071a b9828d2693b14066b0b307719012f33a 40fe19ba2e5e482ba90eb7e7531c14cd 82ea4e6e5db74eb396be0f0471c47e3c 301106135c22461b96fcc783271ad7d0 24e8a0d5176e4c75ac56b3be2394d42a bb313ceee0e24b2199aa191744722f66 047aaba883524149a50b38576ad25a25 a0643fa1dba04fe7ac6966c150ac74ff

# gamma vergleich: b9828d2693b14066b0b307719012f33a 82ea4e6e5db74eb396be0f0471c47e3c 24e8a0d5176e4c75ac56b3be2394d42a

# trace vergleich: b9828d2693b14066b0b307719012f33a 40fe19ba2e5e482ba90eb7e7531c14cd

# den besten mit ein paar anderen : b74822cfa4d04135bee93e3e8b56f633 e330eeac7e7244b4b2bdd2cb1e9c30cb cca29069662141e8881baa0639e85b03 dc4d6525687f4d948fd46e5512ab887d c8a3a7b3736d44a4a23c68929117919c 765138b6ef7f4a5d83ae0f9ae5134e5f 60576dad0a2145ecbfcb8d6f8bb94bab

# e330eeac7e7244b4b2bdd2cb1e9c30cb cca29069662141e8881baa0639e85b03 dc4d6525687f4d948fd46e5512ab887d c8a3a7b3736d44a4a23c68929117919c"