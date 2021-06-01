import mlflow
import os

from dotenv import load_dotenv

# load environment variables from .env
load_dotenv()

# example params
params = {}
params['gamma'] = 0.99
params['alpha'] = 0.0001
params['training_episodes'] = 100


if __name__ == "__main__":
    with mlflow.start_run() as run:
        print("Running mlflow_tracking.py")

        # log params
        mlflow.log_param('training_episodes', params['training_episodes'])
        mlflow.log_param('gamma', params['gamma'])
        mlflow.log_param('alpha', params['alpha'])

        # create test file if doesn't exist
        if not os.path.exists("outputs"):
            os.makedirs("outputs")
            with open("outputs/test.txt", "w") as f:
                f.write("hello world!")

        # log file
        mlflow.log_artifacts("outputs")
