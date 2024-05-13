# Style Guide for Python Code

## 1 Introduction

This document gives coding conventions for the Python code comprising [OpenVINO-XAI (OV-XAI)](../../README.md).

This style guide supplements the [PEP 8 -- Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/)
with a list of *dos and don'ts* for Python code. If no guidelines were found in this style guide then
the [PEP 8 -- Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/) should be followed.

## 2 Automating Code Formatting

To maintain consistency and readability throughout the codebase, we use a set of tools for formatting.
Before committing any changes, it's important to run a pre-commit command to ensure that the code is properly formatted.
You can use the following commands for this:

```bash
pre-commit run --all-files
```

Also recommend configuring your IDE to run Black and isort tools automatically when saving files.

Automatic code formatting is mandatory for all Python files, but you can disable it for specific cases if required.

## 3 Python Language Rules

TBD

## 4 Python Style Rules

TBD
