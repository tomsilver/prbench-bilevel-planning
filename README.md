# Bilevel Planning Baselines for PRBench

![workflow](https://github.com/tomsilver/prbench-bilevel-planning/actions/workflows/ci.yml/badge.svg)

## Installation

1. Recommended: create and source a virtualenv (perhaps with [uv](https://github.com/astral-sh/uv))
2. Clone this repo with submodules: `git clone --recurse-submodules git@github.com:tomsilver/prbench-bilevel-planning.git` 
3. Install this repo: `pip install -e ".[develop]"`
4. Install the submodules:
    - `pip install -e third-party/bilevel-planning`
    - `pip install -e third-party/prbench`
