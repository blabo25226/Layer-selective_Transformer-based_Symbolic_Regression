#!/bin/bash
#install ellyn

git clone  https://github.com/cavalab/ellyn

cd ellyn
# fix version
git checkout cdff25b2851d942db1cdb2a6796ea61c41396c7c

# Ellyn needs compatibility macros with modern Boost at compile time.
sed -i '1i #define BOOST_TIMER_ENABLE_DEPRECATED\n#define BOOST_ERROR_CODE_HEADER_ONLY' src/ellen/stdafx.h

python setup.py install
