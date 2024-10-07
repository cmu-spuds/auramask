# ruff: noqa: E402
import os

os.environ["KERAS_BACKEND"] = "torch"

import auramask
import wandb

if __name__ == "__main__":
    api = wandb.Api()
    artifact: wandb.Artifact = api.artifact(
        "spuds/auramask/run_f2b49943339769994f7166ad3bc10df7_model:latest", type="model"
    )
    weights = artifact.download()
    weights = os.path.join(weights, "76-1.01.weights.h5")
    run = artifact.logged_by()
    config = run.config
    model = auramask.AuraMask(config, weights)
    model.summary()
