import os

from azureml.core import Datastore, Environment, Experiment, Workspace
from azureml.core.compute import ComputeTarget
from azureml.core.dataset import Dataset
from azureml.core.runconfig import RunConfiguration
from azureml.data import OutputFileDatasetConfig
from azureml.data.data_reference import DataReference
from azureml.pipeline.core import Pipeline, PipelineData, PipelineParameter
from azureml.pipeline.steps import PythonScriptStep

from packages.azureml_functions import get_ws


def get_pipeline_steps(ws):
    steps = []

    # Step 1
    df = ws.datasets.get("pump_sensor")

    preprocessing_env = "environments/preprocessing_environment.yml"
    location_steps = "deployment/steps/"
    pipeline_preprocessing_run_config = RunConfiguration()

    env_prep = Environment.from_conda_specification(
        name="pipeline_environment_preprocess",
        file_path=preprocessing_env,
    )
    pipeline_preprocessing_run_config.environment = env_prep
    preprocessing_cluster = ComputeTarget(
        workspace=ws, name="general-8gb")

    train_folder = PipelineData("train_folder")

    preprocessing_step = PythonScriptStep(
        name="Preprocess Data",
        source_directory=location_steps,
        script_name="preprocessing.py",
        inputs=[df.as_named_input("pump_sensor")],
        outputs=[train_folder],
        arguments=["--train_folder", train_folder],
        compute_target=preprocessing_cluster,
        runconfig=pipeline_preprocessing_run_config,
        allow_reuse=True,
    )

    steps.append(preprocessing_step)

    # Step 2
    training_env = "environments/training_environment.yml"
    pipeline_training_run_config = RunConfiguration()

    env_prep = Environment.from_conda_specification(
        name="pipeline_environment_preprocess",
        file_path=training_env,
    )

    pipeline_training_run_config.environment = env_prep
    training_cluster = ComputeTarget(
        workspace=ws, name="memory-optimized")

    prediction_folder = PipelineData("prediction_folder")

    train_step = PythonScriptStep(
        name="Training",
        source_directory=location_steps,
        script_name="training.py",
        inputs=[train_folder],
        outputs=[prediction_folder],
        arguments=["--train_folder", train_folder,
                   "--prediction_folder", prediction_folder],
        compute_target=training_cluster,
        runconfig=pipeline_training_run_config,
        allow_reuse=True,
    )
    steps.append(train_step)

    return steps


if __name__ == "__main__":
    PIPELINE_NAME = "WaterPumpSensors"

    ws = get_ws("train")
    pipeline_steps = get_pipeline_steps(ws)

    pipeline = Pipeline(workspace=ws, steps=pipeline_steps)
    print("Pipeline is built.")

    # Create an experiment and run the pipeline
    # repo_shorthand = Repository('.').head.shorthand
    repo_shorthand = "train"
    name = repo_shorthand + "_" + PIPELINE_NAME  # branch name
    name = name.replace("/", "_")

    experiment = Experiment(workspace=ws, name=name)
    pipeline_run = experiment.submit(pipeline)
    print("Pipeline submitted for execution.")
    pipeline_result = pipeline_run.wait_for_completion(show_output=True)


# if __name__ == "__main__":
#     PIPELINE_NAME = "water_pump_simulation"

#     ws = get_ws("train")
#     # Get the default datastore
#     datastore = ws.get_default_datastore()

#     pipeline_runs = []

#     for filename in file_names:
#         # date_string = filename[18:26]

#         pipeline_steps = get_pipeline_steps(ws)

#         pipeline = Pipeline(workspace=ws, steps=pipeline_steps)
#         print("Pipeline is built.")

#         name = f"ravago-train_{PIPELINE_NAME}"

#         experiment = Experiment(workspace=ws, name=name)
#         pipeline_run = experiment.submit(pipeline)
#         print("Pipeline submitted for execution.")
#         pipeline_runs.append(pipeline_run)
#         # pipeline_result = pipeline_run.wait_for_completion(show_output=True)
#         # Wait for all pipeline runs to complete
#     for run in pipeline_runs:
#         run.wait_for_completion(show_output=True)
