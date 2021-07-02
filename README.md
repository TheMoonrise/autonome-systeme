# **Practical Course Autonome Systeme**

This repository holds the project files for team **Machine Learning League (MLL)**

# Unity Domain

Our work is based on the [Crawler Unity ML Domain](https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Learning-Environment-Examples.md).
The domain features a four legged creature whose task it is to collect blocks, which are randomly placed within the arena.
There is always one block active in the arent. When this block is collected a new block appears.
Rewards are issued for moving towards the block while looking at it.
The overvation space consists of 158 continuous inputs while the actionspace holds 20 continuous outputs, which map to the target rotation of the joints within the creature.
For our experiments we did not modify the rewards or other domain properties which could impact receiving rewards.
This allows us to compare our results to other crawler implementation which use the same version of the domain.

# Team Contributions

The following listing gives a brief overview of the project contributions of each team member.

## Verena

-   Implementation Algorithm SAC
-   Hyperparameter Exploration & Training
-   Evaluation of Training Results

## Mariam

-   Implementation Algorithm SAC
-   Hyperparameter Exploration & Training
-   Evaluation of Training Results

## Antonia

-   Tracking of Metrics (MLflow)
-   Plotting & Data Visualization
-   Hyperparameter Exploration & Training
-   Evaluation of Training Results

## Severin

-   Setup & Maintanance Server Infrastructure
-   Hyperparameter Exploration & Training
-   Evaluation of Training Results

## Patrick

-   Implementation Algorithm PPO
-   Configuration & Customization Unity Domain
-   Hyperparameter Exploration & Training
-   Evaluation of Training Results

# Instructions

The following section describes how to run the unity domain.

## Commandline Arguments

Many scripts support the same commandline arguments for setting training parameters or modifying the domain.
Below is a listing of all arguments alongside an explanation.

#### `--model <model name>`

The pretrained model to be loaded.
Models are loaded from the `models\ppo` or `models\sac` directories.
Trained versions of both algorithms can be found there.

#### `--params <params json file>`

The parameter file to be loaded. This works similar to the model argument.
Parameter files are loaded from the `params` directories.

#### `--runs <int>`

The number of consecutive runs to be performed with the given settings.

#### `--speed <float>`

The speed of the Unity simulation. The default value is 1. Higher numbers increase the simulation and thus training speed at the cost of accuracy.

#### `--quality <int>`

The quality of the Unity simulation. This also affects the visual quality of the rendered view.
For best viewing quality set the value to be 5 or higher.

#### `--slipperiness <float>`

The slipperiness of the ground in the simulation. Values must be between 0 and 1. Default is 0.

#### `--steepness <float>`

The roughness of the terrain in the simulation. Values must be positive. Default is 0.
For reasonable results do not use values above 1.

#### `--hue <int>`

The color hue of the crawler. Values must be between 0 and 360.

#### `--no--window`

Hides the simulation window.

#### `--no-mlflow`

Does not log the run to the mlflow server. This is usefull for debugging as well as when the mlflow server is not available.

---

## Baseline

The baseline agent can be run by executing one of the builds in the `environment` directory.
For windows execute `environments\windows\Crawler.exe`.

Additionaly the commandline arguments `--slipperiness --steepness --hue` can be passed to run the baseline agent with different terrain configurations.

---

## PPO Playback

To show the ppo agent in action without training run `main\play_crawler_ppo.py`.

This supports the commandline arguments `--model --params --speed --quality --slipperiness --steepness --hue`.

Note: Make sure the parameter file net size matches the net size of the model argument.

An example call to run one of the more advanced ppo models would be:

```
python main\play_crawler_ppo.py --model nancy-v10 --params small_net.json --quality 5
```

## PPO Training

To train a model using ppo run `main\train_crawler_ppo.py`.

This supports the commandline arguments `--model --params --runs --speed --quality --slipperiness --steepness --hue --no-window --no-mlflow`.

Note: Make sure the parameter file net size matches the net size of the model argument.

An example call to train a new ppo model would be:

```
python main\train_crawler_ppo.py --params small_net.json --speed 4 --no-window --no-mlflow
```

---

## SAC Playback

To show the ppo agent in action without training run `main\play_crawler_sac.py`.

This supports the commandline arguments `--model --params --speed --quality --slipperiness --steepness --hue`.

Note: Make sure the parameter file net size matches the net size of the model argument.

An example call to run one of the more advanced sac models would be:

```
python main\play_crawler_sac.py --model name --params params.json
```

## SAC Training

To train a model using ppo run `main\train_crawler_sac.py`.

This supports the commandline arguments `--model --params --runs --speed --quality --slipperiness --steepness --hue --no-window --no-mlflow`.

Note: Make sure the parameter file net size matches the net size of the model argument.

An example call to train a new sac model would be:

```
python main\train_crawler_sac.py --params params.json --no-window --no-mlflow
```

# Models & Hyper Parameters

The following section provides the hyper parameters used for training some of the final models.

Note that some models were trained with varying parameters. If this was the case the most common parameters will be listed here.

## PPO 128x128 "Nancy"

This model uses two layers of 128 nodes each. Additionally this model uses batch normalization for each step.

The average performance of the 10th version of this model `nancy-v10` was at 900.

```
clip                0.2
update epochs       1
influence critic    0.7
influence entropy   0.01
gamma               0.995
lmbda               1
collected trace     32
learning rate       2e-5
hidden layer 01     128
hidden layer 02     128
```