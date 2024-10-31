from databricks.model_training import foundation_model as fm
from nodes.import_export.databricks_base import DatabricksBase
from databricks.sdk import WorkspaceClient
import mlflow
import time
import os
import re

mlflow.set_registry_uri("databricks-uc")


def ensure_cluster_running(host, token, cluster_id):
    client = WorkspaceClient(
        host=host,
        token=token
    )

    # Loop until the cluster is in 'RUNNING' state
    while True:
        cluster_info = client.clusters.get(cluster_id)

        if cluster_info.state.name == 'RUNNING':
            print("Cluster is now running.")
            break
        elif cluster_info.state.name in ['PENDING', 'RESTARTING']:
            print("Cluster is starting. Current state:", cluster_info.state.name)
        else:
            # Cluster is not running, attempt to start it
            print("Starting cluster. Current state:", cluster_info.state.name)
            client.clusters.start(cluster_id)

        # Wait before checking the state again
        time.sleep(10)


def run_train(model_name, catalog, db, token, host, cluster_id):
    registered_model_name = f"{catalog}.{db}.classify_" + re.sub(r'[^a-zA-Z0-9]', '_',
                                                                 model_name)  # TODO: not sure about that
    train_data_path = f'{catalog}.{db}.training_dataset'

    os.environ["DATABRICKS_HOST"] = host
    os.environ["DATABRICKS_TOKEN"] = token

    # # # Define the experiment path
    # experiment_path = f"/Users/{user}/{run_name}"  # TODO: not used for now

    # Create the experiment if it doesn't exist
    # if not mlflow.get_experiment_by_name(experiment_path):
    #     mlflow.create_experiment(experiment_path)
    # mlflow.set_experiment(experiment_path)

    # Check that cluster is running
    ensure_cluster_running(host=host, token=token, cluster_id=cluster_id)

    attempts = 3
    for attempt in range(attempts):
        try:
            run = fm.create(
                data_prep_cluster_id=cluster_id,
                model=model_name,
                # experiment_path='/Users/roni@azure.dataloop.ai/testing-fine-tuning',
                train_data_path=train_data_path,
                task_type="CHAT_COMPLETION",
                training_duration="10ep",  # only 10 epochs for the demo
                register_to=registered_model_name,
                learning_rate=5e-7,
            )  # your creation parameters
            break
        except TimeoutError:
            print(f"Attempt {attempt + 1}/{attempts} timed out. Retrying...")
            time.sleep(10)  # Wait a bit before retrying


if __name__ == '__main__':
    import dotenv

    dotenv.load_dotenv()

    catalog = "datakoop_poc"
    schema = "ludo_test"
    table_name = "roni-test-table"
    cluster_id = "1031-091041-lezk7mgo"
    base_model_name = "meta-llama/Meta-Llama-3-8B"

    run_train(model_name=base_model_name,
              catalog=catalog,
              db=schema,
              token=os.environ.get("token"),
              host=os.environ.get("server_hostname"),
              cluster_id=cluster_id)
