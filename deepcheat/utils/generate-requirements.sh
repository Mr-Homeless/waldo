#!/bin/bash
cd ..
conda env export --from-history > environment.yml
sed -i '$ d' environment.yml