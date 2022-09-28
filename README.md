# interpretableAI_DRP
## Overview
The code for running each model is divided into individual sub-folders. Two types of model execution can be done: 1) run a pretrained model with specified hyperparameters; 2) run a model from scratch with specified hyperparameters. The former execution can be done by running the *run_pretrained.sh* script and the latter can be done by running *run_model_with_hyp.sh* script.

Hyperparameter tuning has been performed on the validation set and the best set of hyperparameters for each validation strategy (leave-ccls-out/LCO, leave-drugs-out/LDO, leave-pairs-out /LPO) and each pathway collection (KEGG, PID, Reactome) are provided in sub-folders named *best_hyp*. 

All pathway-based models (PathDNN, ConsDeepSignaling, HiDRA, PathDSP) are re-implementations of the original models, with a very small component of code being adaptations (direct usage) of the original code provided by the authors of these pathway-based models. References for such adaptations are included in the comments of the code. 

## References
- **PathDNN**
[Link to PathDNN paper](https://pubs.acs.org/doi/10.1021/acs.jcim.0c00331)
Deng, L. et al. Pathway-guided deep neural network toward interpretable and predictive modeling of drug sensitivity. J. Chem. Inf. Model. 60, 4497–4505 (2020)

- **ConsDeepSignaling (CDS)**
[Link to ConsDeepSignaling paper](https://www.frontiersin.org/articles/10.3389/fbinf.2021.639349/full)
Zhang, H., Chen, Y. & Li, F. Predicting Anticancer Drug Response With Deep Learning Constrained by Signaling Pathways. Front. Bioinforma. 1, (2021)

- **HiDRA**
[Link to HiDRA paper](https://pubs.acs.org/doi/10.1021/acs.jcim.1c00706)
Jin, I. & Nam, H. HiDRA: Hierarchical Network for Drug Response Prediction with Attention. Journal of Chemical Information and Modeling vol. 61 3858–3867 (2021)

- **PathDSP**
[Link to PathDSP paper](https://www.nature.com/articles/s41598-021-82612-7)
Tang, Y. C. & Gottlieb, A. Explainable drug sensitivity prediction through cancer pathway enrichment. Sci. Rep. 11, (2021)

## Input Data
The input data for the models can be found at (Zenodo link).

## Environment Requirement
- python -3.9.7
- pytorch -1.11.0
- pandas -1.3.4
- numpy -1.20.3
- scipy -1.7.1
- scikit-learn -0.24.2

## Usage

 - Download the input data folder from (Zenodo link) and place the input
   data folder under the **same directory** as the *models* folder (found in this GitHub repository). You could place the input data folder else where, but make sure the dataroot argument in the bash scripts is modified accordingly. **DO NOT MODIFY** the input data folder.
 - To run any desired model, open the terminal and navigate into the corresponding model folder (e.g. `cd models/PathDNN/`) 
 - Depending on what task you wish to run, modify the arguments (dataroot, outroot, etc.) in *run_pretrained.sh* or *run_model_with_hyp.sh* accordingly.
 - Activate the Python environment you intend to use in the terminal.
 - If you wish to run a **pretrained model** with specified hyperparameters, run `bash run_pretrained.sh` in the terminal; if you wish to run a **model from scratch** using specified hyperparameters, run `bash run_model_with_hyp.sh`
