# PLMF-MDA
 Submitted to journal Advanced Science
## 1. Overview
The code for paper "A pre-trained language model-based cross-modal fusion framework for predicting miRNA-drug resistance and sensitivity associations". The repository is organized as follows:

+ `data/MDR/MDS` contains the data in the paper:
  * `miRNA_drug_R_matrix.xlsx/miRNA_drug_S_matrix.xlsx` contains known miRNA-drug resistance and sensitivity associations;
  * `drug_ID_R.xlsx/drug_ID_S.xlsx` contains drug ID, name, smiles ;
  * `miRNA_ID_R.xlsx/miRNA_ID_S.xlsx` contains miRNA ID, name, sequences ;
  * `pos_MDA_R.edgelist/pos_MDA_S.edgelist` contains positive sample pairs.
  * `neg_MDA_R.edgelist/pos_MDA_S.edgelist` contains negative sample pairs.
    
+ `code/`
  * `parms_setting.py`contains hyperparmeters;
  * `data_preprocess.py` contains the preprocess of data;
  * `layer.py` contains MGCNA's model layer;
  * `train.py` contains training and testing code;
  * `DeepChem` contains ChemBERTa files;
  
## 2. Dependencies
* numpy == 1.24.4
* torch == 2.1.2+cu118
* sklearn == 1.3.0
* torch-geometric == 2.4.0

## 3. Quick Start
Here we provide a example:

1. Download and upzip our data and code files
2. Run "train.py" 

## 4. Reminder
The dataset was partitioned into training and testing sets with an 8:2 ratio.

## 5. Contacts
If you have any questions, please email Nan Sheng (shengnan@jlu.edu.cn)
