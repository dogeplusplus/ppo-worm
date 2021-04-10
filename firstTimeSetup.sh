#!/bin/bash
set -xeuf -o pipefail

rm -rf venv

python -m venv venv
source venv/bin/activate

pip install setuptools==40.3.0 pip==20.1 pytest
pip install -r requirements.txt
