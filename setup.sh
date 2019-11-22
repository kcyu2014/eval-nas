#!/usr/bin/env bash

conda install pytorch=0.3 cuda90 -c pytorch -y
conda install scipy ipython jupyter pandas plotly -y
pip install tensorboardX ipdb