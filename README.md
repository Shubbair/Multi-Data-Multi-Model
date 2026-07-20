# Multi-data Multi-Model Ensemble

combining multi data presentation and using multi-model ensembel(average, meta-learner) to produce robust result.

**Inspired by human(brain) multi-level of attention.**

## Data pipeline generation

from raw images dataset of Alphabet Sign lanaguge *(_rgb.zip)*, extract edges and skeleton key-points.

<img src="assets/data_pipeline.png" width="25%"/>

unzip dataset file then , generate data

code file : 
`edge_data.py`
`keypoint_data.py`

## Model Architecture

<img src="assets/model_architecture.png" width="25%"/>

Modeling : `multi-view-modeling.ipynb`

using cross-validation to get rid of data leakage problem, result : </br>

<img src="assets/conf-matrix.png" width="80%"/>
