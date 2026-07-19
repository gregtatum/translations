# Publish orchestrator

Publishing is a multi-step process that is fallible across multiple package managers. Here we want to be in lockstep on the release. The goal would be to update the publish script to be a better orchestrator. Let's do a command and control interface that hides stdout, and only surfaces stdout/err when there is a failure. Take inspiration from `task rs:check`. It should show the current status of the publish. So if a previous one failed, it should surface that so I can run it over and over again until I have a pass. So maybe we can kick off a `patch` `minor` `major` run, then re-run as much as possible with a command and control interface that suggests self-healing. That way for instance, if I'm not logged into npm, it doesn't matter, I can just re-run it following the instructions until I get a full success. On failures I get the full log, and helpful hints on what happens next.

Consider how to do this for:

 - crates.io fxtranslate
 - crates.io fxtranslate-cli
 - npm fxtranslate
 - pypi fxtranslate
    - all platform wheels

Consider ergonomics and style for how it looks based off of task rs:check.
