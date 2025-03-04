# Re-using Old Runs

## `experiments.previous_group_ids`

Sometimes you want to re-use tasks from old runs. In this case you can include previous
group ids. These graphs will be traversed attempting to fill in all of the tasks before
the `experiments.start_stage`. See `valid-stages` in [taskcluster/config.yml](https://github.com/mozilla/translations/blob/main/taskcluster/config.yml)
for the valid stages.

```yaml
experiments:
  start-stage: train-student
  previous-group-ids:
    - Fzrv_XPwTs6KVNkktUeuOA
    - YoeLXLcMTyKw4hVrIwLPiQ
```

## `experiments.existing-tasks`

Explicit tasks can be re-used from older runs. This can help with `previous_group_ids`
fails, or you need an explicit override.

```yaml
experiments:
  existing_tasks: {
    "build-docker-image-base": "BAvLUilqQ3SYqy6Ck55CUQ",
    "build-docker-image-test": "f0gbptvMTDaKODjqL9hlOw",
    "build-docker-image-toolchain-build": "LlZa8-L9TRemgyzQcAxuHw",
    "build-docker-image-train": "fBMJa9R5SKaXd2wgWeD5yQ",
    "fetch-browsermt-marian": "BRviRlEMTie8AUFf5prHvg",
  }
```

## `experiments.target-stage`

In addition to re-using old tasks there can be a target stage, and no tasks past that
target will be run. See `valid-stages` in [taskcluster/config.yml](https://github.com/mozilla/translations/blob/main/taskcluster/config.yml)
for the valid stages.

```yaml
experiments:
  start-stage: train-student
  target-stage: evaluate-student
  previous_group_ids:
    - Fzrv_XPwTs6KVNkktUeuOA
    - YoeLXLcMTyKw4hVrIwLPiQ
```
