# Firefox Translations training
Training pipelines for Firefox Translations machine translation models.

The trained models are hosted in [firefox-translations-models](https://github.com/mozilla/firefox-translations-models/) repository,
compatible with [bergamot-translator](https://github.com/mozilla/bergamot-translator) and
power the Firefox web page translation starting with version 118.

The pipeline was originally developed as a part of [Bergamot](https://browser.mt/) project  that focuses on improving client-side machine translation in a web browser.

[Documentation](https://mozilla.github.io/firefox-translations-training/)

## Contributing

If you would like to contribute to the project there are various ways to help out.

## Request a language

If you would like to request a new language you can do so on [Mozilla Connect](https://connect.mozilla.org). Search for your language, and thumbs it up, or if it's not there, make a new post.

## Evaluate our models

We are always looking for better qualitative understanding of our language models, which means you can try out our models and leave feedback on our [Firefox Translations Matrix chat channel](https://chat.mozilla.org/#/room/#firefoxtranslations:mozilla.org) or in GitHub issues. We also may have questions about specific languages, and if you leave your name and the language you can help with, we may reach out.

## Researchers for Machine Learning

If you are a researcher and are interested in using our pipeline, you can also [reach out](https://chat.mozilla.org/#/room/#firefoxtranslations:mozilla.org).
We have optimized the pipeline to run on Mozilla cloud resources via [Taskcluster](https://taskcluster.net/) and no longer actively maintain the Snakemake configuration, but we do accept PRs to fix it. If you can help contribute specific training configs and data

We can always use language experts to help our understanding of cleaning rules, segmentation issues, and other language specific issues.
If you are interested on hacking on something and contributing some code, check out the GitHub issues.

## Pipeline

The pipeline is capable of training a translation model for a language pair end to end.
Translation quality depends on the chosen datasets, data cleaning procedures and hyperparameters.
Some settings, especially low resource languages might require extra tuning.

We use the fast translation engine [Marian](https://marian-nmt.github.io).

You can find more details about the pipeline steps in the [documentation](docs/pipeline-steps.md).

## Orchestrators

An orchestrator is responsible for workflow management and parallelization.

- [Taskcluster](https://taskcluster.net/) - Mozilla task execution framework. It is also used for Firefox CI.
  It provides access to the hybrid cloud workers (GCP + on-prem) with increased scalability and observability.
  [Usage instructions](docs/task-cluster.md).
- [Snakemake](https://snakemake.github.io/) - a file based orchestrator that allows to run the pipeline locally or on a Slurm cluster.
  [Usage instructions](docs/snakemake.md). (The integration is not maintained since Mozilla has switched to Taskcluster. Contributions are welcome.)

## Experiment tracking

Marian training metrics are parsed from logs and published using a custom module within the `tracking` directory.
More information is available [here](docs/tracking.md).

## Learning resources

- High level overview [post on Mozilla Hacks](https://hacks.mozilla.org/2022/06/training-efficient-neural-network-models-for-firefox-translations/)

- [Model training guide](docs/training-guide.md) - practical advice on how to use the pipeline
- [Reference papers](docs/references.md)


## Acknowledgements
This project uses materials developed by:
- Bergamot project ([github](https://github.com/browsermt), [website](https://browser.mt/)) that has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 825303
- HPLT project ([github](https://github.com/hplt-project), [website](https://hplt-project.org/)) that has received funding from the European Union’s Horizon Europe research and innovation programme under grant agreement No 101070350 and from UK Research and Innovation (UKRI) under the UK government’s Horizon Europe funding guarantee [grant number 10052546]
- OPUS-MT project ([github](https://github.com/Helsinki-NLP/Opus-MT), [website](https://opus.nlpl.eu/))
- Many other open source projects and research papers (see [References](docs/references.md))
