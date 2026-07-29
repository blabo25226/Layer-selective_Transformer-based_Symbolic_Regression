#!/bin/bash
#install ellyn

git clone  https://github.com/cavalab/ellyn

cd ellyn
# fix version
git checkout cdff25b2851d942db1cdb2a6796ea61c41396c7c

# Ellyn depends on deprecated Boost timer headers on recent Boost releases.
sed -i '1i #define BOOST_TIMER_ENABLE_DEPRECATED\n#define BOOST_ERROR_CODE_HEADER_ONLY' src/ellen/stdafx.h

python setup.py install
