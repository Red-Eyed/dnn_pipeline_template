[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/Python-%3E%3D3.11-blue)]()
[![Build Status](https://img.shields.io/github/actions/workflow/status/<your-org>/<your-repo>/ci.yml?branch=main)]()

# DNN Pipeline Template

A starter repository for deep‐learning projects, providing a standardized environment, launcher script, and tooling for debugging & profiling.

---

## Table of Contents

1. [Features](#features)  
2. [Prerequisites](#prerequisites)  
3. [Project Setup](#project-setup)  
4. [Usage](#usage)  
   - [Launching a Script](#launching-a-script)  
   - [Debug Mode](#debug-mode)  
   - [Profiling Mode](#profiling-mode)  
5. [Directory Layout](#directory-layout)  
6. [Contributing](#contributing)  
7. [License](#license)  

---

## Features

- **Portable launcher** (`start.py`) isolates your system environment  
- **PYTHONPATH** is automatically set to project root  
- Single entry point for scripts, debug, and profiling  
- Git LFS support for large model/checkpoint files  

---

## Prerequisites

- Ubuntu 24.04 (other OS not officially supported)  
- Python 3.11+  
- Git with [Git LFS](https://git-lfs.github.com/)  

---

## Project Setup

```bash
# 1. Clone & fetch LFS objects
git clone https://github.com/<your-org>/<your-repo>.git
cd <your-repo>
git lfs install
git lfs pull

# 2. Create & activate virtualenv, install dependencies
./install_env.sh
````

---

## Usage

All project scripts are launched via the `start.py` wrapper:

```bash
./start.py -- /path/to/your_script.py [--script-args]
```

* `start.py` ensures a clean virtualenv and sets `PYTHONPATH` to the repo root
* Use `--` to separate `start.py` options from your script’s arguments

### Launching a Script

```bash
./start.py -- scripts/train.py -- --config configs/train.yaml
```

### Debug Mode

Waits for a VS Code debug client to attach on port 5678:

```bash
./start.py -d -- scripts/train.py -- --config configs/train.yaml
```

1. Press F5 in VS Code
2. Select the “Attach to start.py” debug configuration

### Profiling Mode

Profiles with [Scalene](https://github.com/plasma-umass/scalene), output to `prof.json`:

```bash
./start.py -p -- scripts/train.py -- --config configs/train.yaml
```

---

## Directory Layout

```
├── install_env.sh       # creates virtualenv, installs dependencies
├── start.py             # launcher + debug/profiler wrapper
├── requirements.txt     # Python dependencies
├── scripts/             # user scripts (train.py, eval.py, etc.)
├── configs/             # YAML/JSON config files
├── src/                 # project code (models/, data/, utils/)
└── prof.json            # latest Scalene profile output
```

---

## Contributing

1. Fork the repo
2. Create a feature branch (`git checkout -b feat/my-feature`)
3. Commit your changes
4. Open a pull request against `main`

Please follow PEP 8, include tests in `tests/`, and ensure all CI checks pass.

---

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

```

