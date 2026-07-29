# !/bin/bash

git clone https://github.com/dso-org/deep-symbolic-optimization.git

cd deep-symbolic-optimization

# latest version. pypi version is broken
pip install git+https://github.com/DEAP/deap.git@master

pip install -e ./dso # Install DSO package and core dependencies
