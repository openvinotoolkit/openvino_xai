# Contributing to OpenVINO™ Explainable AI (XAI) Toolkit

## Code style

Changes to OpenVINO XAI Python code should conform to [Python Style Guide](./docs/styleguide/PyGuide.md)

Basic code style and static checks are enforced using a `pre-commit` Github action.
The exact checks that are run are described in the corresponding [config file](./.pre-commit-config.yaml).
The checks can be run locally using `pre-commit run --all-files` or `pre-commit run --files FILENAME`.
Most of these checks do in-place fixes at the same time they are run using `pre-commit run --all-files`
