from azureml.core import Workspace
from azureml.core.authentication import InteractiveLoginAuthentication


def get_ws(stage="ml"):
    stages = {"dev", "train"}
    if stage not in stages:
        raise ValueError(
            "Invalid stage for workspace: got %s, should be from %s" % (stage, stages))

    if stage in {"dev", "train"}:
        print("Logging in as user")
        credential = InteractiveLoginAuthentication()

    config_path = ".cloud/.azure/config_{stage}.json".format(
        stage=stage.upper())
    ws = Workspace.from_config(config_path, auth=credential)

    return ws
