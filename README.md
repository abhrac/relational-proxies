# Relational Proxies: Emergent Relationships as Fine-Grained Discriminators

Official implementation of "Relational Proxies: Emergent Relationships as Fine-Grained Discriminators",
NeurIPS 2022.

Our framework helps learn a cross-view representation by modelling local-to-global emergent relationships
for Fine-Grained Visual Categorization (FGVC).

![Model Diagram](./assests/relational_proxies_diagram.png)

## Environment Setup

This project is implemented using PyTorch. A conda environment with all related dependencies can be created as follows:
1. Clone the project repository:
```shell
git clone https://github.com/abhrac/relational-proxies.git
cd relational-proxies
```
2. Create and activate conda environment:
```shell
conda env create -f environment.yml
conda activate environment.yml
```
3. Download the .pth file from
[here](https://drive.google.com/file/d/1P556ct4WTxWgZSLsKj4k9PZ52g6StGFA/view?usp=sharing)
and place it in the `./view_localizer/` folder under the project root.

## Training
To train the model from scratch, run the following:
```shell
python3 src/main.py --data_root='RootDirOfAllDatasets' --dataset='DatasetName'
```
The `run_expt.sh` file contains sample training commands.

## Evaluation
To evaluate on a dataset using pretrained weights, first download the model for the corresponding dataset from
[here](https://drive.google.com/drive/folders/1WR9qqFmhArHJqg78wsffhQtAbiW3V77R?usp=sharing)
and place it under the folder `./checkpoint/$DataSetName/`,
where `./checkpoint` is under the project root, but could optionally be elsewhere too
(see `src/options.py`). Then, run the following command:
```shell
python3 src/main.py --data_root='RootDirForAllDatasets' --dataset='DatasetName' --pretrained --eval_only
```

## Citation
```
@inproceedings{Chaudhuri2022RelationalProxies,
 author = {Abhra Chaudhuri, Massimiliano Mancini, Zeynep Akata, Anjan Dutta},
 booktitle = {Proceedings of Advances in Neural Information Processing Systems (NeurIPS)},
 title = {Relational Proxies},
 year = {2022}
}
```
