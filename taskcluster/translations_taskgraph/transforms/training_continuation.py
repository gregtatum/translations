from dataclasses import dataclass
from enum import Enum
from typing import Generator, Literal
from taskgraph.transforms.base import TransformSequence
from urllib.parse import urljoin
import os

"""
Transform jobs to be able to use pre-trained models.
"""

CONTINUE_TRAINING_ARTIFACTS = (
    "devset.out",
    "model.npz",
    "model.npz.best-bleu-detok.npz",
    "model.npz.best-bleu-detok.npz.decoder.yml",
    "model.npz.best-ce-mean-words.npz",
    "model.npz.best-ce-mean-words.npz.decoder.yml",
    "final.model.npz.best-chrf.npz",
    "model.npz.best-chrf.npz",
    "final.model.npz.best-chrf.npz.decoder.yml",
    "model.npz.best-chrf.npz.decoder.yml",
    "model.npz.decoder.yml",
    "model.npz.optimizer.npz",
    "model.npz.progress.yml",
    "model.npz.yml",
    "train.log",
    "valid.log",
    "vocab.spm",
)

INITIALIZE_MODEL_ARTIFACTS = (
    "model.npz.best-bleu-detok.npz",
    "model.npz.best-ce-mean-words.npz",
    "final.model.npz.best-chrf.npz",
    "model.npz.best-chrf.npz",
)

ModelMode = Enum(
    "Mode",
    [
        "init",
        "continue",
        "use",
    ],
)


@dataclass
class PretrainedModel:
    """
    The YAML object that represents a pre-trained model.
    """

    urls: list[str]
    mode: ModelMode
    # In the future "opusmt" may be supported.
    type: Literal["npz"]

    def get_artifact_names(self):
        artifacts = {
            "init": INITIALIZE_MODEL_ARTIFACTS,
            "continue": CONTINUE_TRAINING_ARTIFACTS,
            "use": CONTINUE_TRAINING_ARTIFACTS,
        }
        return artifacts[self.mode]


def get_artifact_mounts(
    urls: list[str], directory: str, artifact_names: str
) -> Generator[list[dict[any]], None, None]:
    """
    Build a list of artifact mounts that will mount a remote URL file to the tasks local
    file system.

    For instance, given: "https://example.com/en-ru"

    This will download files such as:
      "https://example.com/en-ru/model.npz.best-bleu-detok.npz"
      "https://example.com/en-ru/model.npz.best-ce-mean-words.npz",
      etc.
    """

    for url in urls:
        artifact_mounts = []
        for artifact_name in artifact_names:
            # Ensure the url ends with a "/"
            normalized_url = f"{url}/" if not url.endswith("/") else url
            artifact_mounts.append(
                {
                    "content": {"url": urljoin(normalized_url, artifact_name)},
                    "file": os.path.join(directory, artifact_name),
                }
            )
        yield artifact_mounts


transforms = TransformSequence()


@transforms.add
def add_pretrained_model_mounts(config, jobs):
    """
    The transform for modifying the task graph to use pre-trained models.

    See docs/using-pretrained-models.md
    """

    # Example:
    # pretrained-models:
    #   train-backwards:
    #     urls: [https://storage.googleapis.com/bucket-name/models/ru-en/backward]
    #     mode: "use"
    #     type: "default"
    pretrained_models = config.params["training_config"]["experiment"].get("pretrained-models", {})

    for job in jobs:
        pretrained_model_dict = pretrained_models.get(config.kind, None)
        if pretrained_model_dict:
            pretrained_model = PretrainedModel(**pretrained_model_dict)

            # Add the pretrained artifacts to the mounts.
            mounts = job["worker"].get("mounts", [])
            mounts.extend(
                get_artifact_mounts(
                    urls=pretrained_model.urls,
                    directory="./artifacts",
                    artifact_names=pretrained_model.get_artifact_names(),
                )
            )
            job["worker"]["mounts"] = mounts

            # Remove any vocab training, as this is using a pre-existing vocab.
            job["dependencies"].pop("train-vocab")
            job["fetches"].pop("train-vocab")

            if pretrained_model.mode == "use":
                # In "use" mode, no upstream dependencies of the training job are needed - the
                # task simply republishes the pretrained artifacts.
                job["dependencies"] = {}
                job["fetches"] = {}

                # We also need to adjust the caching parameters. The only thing that should influence
                # the cache digest are the pretrained model parameters.
                job["attributes"]["cache"]["resources"] = []
                job["attributes"]["cache"]["from-parameters"] = {
                    p: v
                    for p, v in job["attributes"]["cache"]["from-parameters"].items()
                    if p.startswith("pretrained")
                }

        yield job
