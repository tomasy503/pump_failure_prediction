import os

from azureml.core import Datastore, Environment, Experiment, Workspace
from azureml.core.compute import ComputeTarget
from azureml.core.dataset import Dataset
from azureml.core.runconfig import RunConfiguration
from azureml.data import OutputFileDatasetConfig
from azureml.data.data_reference import DataReference
from azureml.pipeline.core import Pipeline, PipelineData, PipelineParameter
from azureml.pipeline.steps import PythonScriptStep

from src.utils.azureml_functions import get_ws


def get_pipeline_steps(ws):
    steps = []

    # Step 1
    df = ws.datasets.get("pump_sensor")

    preprocessing_env = "environments/preprocessing_environment.yml"
    location_preprocessing = "src/data/"
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
        source_directory=location_preprocessing,
        script_name="run_preprocessing.py",
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
    location_training = "src/training/"
    pipeline_training_run_config = RunConfiguration()

    env_prep = Environment.from_conda_specification(
        name="pipeline_environment_preprocess",
        file_path=training_env,
    )

    pipeline_training_run_config.environment = env_prep
    training_cluster = ComputeTarget(
        workspace=ws, name="memory-optimized")

    prediction_folder = PipelineData("prediction_folder")
    model_output = PipelineData("model_output")

    train_step = PythonScriptStep(
        name="Training",
        source_directory=location_training,
        script_name="run_training.py",
        inputs=[train_folder],
        outputs=[prediction_folder, model_output],
        arguments=["--train_folder", train_folder, "--prediction_folder",
                   prediction_folder, "--model_output", model_output],
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
    repo_shorthand = "train"
    name = repo_shorthand + "_" + PIPELINE_NAME  # branch name
    name = name.replace("/", "_")

    experiment = Experiment(workspace=ws, name=name)
    pipeline_run = experiment.submit(pipeline)
    print("Pipeline submitted for execution.")
    pipeline_result = pipeline_run.wait_for_completion(show_output=True)
