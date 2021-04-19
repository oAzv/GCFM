# GCFM

A deep fusion model with the input of alignment-based feature representations, for the classification of alignments of pairwise sequences of ncRNAs. This model integrates the convolution module with the cascade forest of GcForest, which can learn abstraction features at high levels, adjust part of model architecture automatically during training, and achieve great performance.

## Abstract

Non-coding RNAs (ncRNAs) play crucial roles in multiple biological processes. However, only a few ncRNAs’ functions have been well studied. Given the significance of ncRNAs classification for understanding ncRNAs’ functions, more and more computational methods have been introduced to improve the classification automatically and accurately. In this paper, based on a convolutional neural network and a deep forest algorithm, multi-grained cascade forest (GcForest), we propose a novel deep fusion learning framework, GcForest fusion method (GCFM), to classify alignments of ncRNA sequences for accurate clustering of ncRNAs. GCFM integrates a multi-view structure feature representation including sequence-structure alignment encoding, structure image representation, and shape alignment encoding of structural subunits, enabling us to capture the potential specificity between ncRNAs. For the classification of pairwise alignment of two ncRNA sequences, the F-value of GCFM improves 6% than an existing alignment-based method. Furthermore, the clustering of ncRNA families is carried out based on the classification matrix generated from GCFM. Results suggest better performance (with 20% Accuracy improved) than existing ncRNA clustering methods (RNAclust, Ensembleclust, and CNNclust). Additionally, we apply GCFM to construct a phylogenetic tree of ncRNA and predict the probability of interactions between RNAs. Most ncRNAs are located correctly in the phylogenetic tree, and the prediction accuracy of RNA interaction is 90.63%. A web server (http://bmbl.sdstate.edu/gcfm/) is developed to maximize its availability, and the source code and related data are available at the same URL.

## Environment

GCFM consists of a series of scripts that can be executed in stages, requires a basic UNIX/Linux environment. The gcc compiler with version 4.8.4 or higher is required. Currently, GCFM does not support Mac or Windows system. Due to the large memory requirement, we recommend users run GCFM on a high-performance computer (HPC), rather than local computers.

## Install

```
# Simplified tree diagram of files
./GCFM_sources
├── requirements.txt
├── scripts
│   ├── CNN
│   ├── gcforest
│   └── WK_NetArch
└── tools-non-py
    ├── dafs
    ├── lib
    │   ├── parse_secondary_structure
    │   └── parse_secondary_structure.cpp
    └── RNAfold
```
Download the source from Download page of web server or:
```
cd /your/working/path/
wget http://bmbl.sdstate.edu/gcfm/static/clf/code/GCFM_sources.zip
```
Unzip the file:
```
unzip GCFM_sources.zip && rm -rf GCFM_sources.zip
```
Install Python packages according to the ```requirements.txt``` manually or: 
```
cd ./GCFM_sources
pip install -r requirements.txt
```
Change the permissions (need to be executable) of ```dafs``` and ```RNAfold``` in directory tools-non-py or:
```
cd ./GCFM_sources/tools-non-py
chmod 777 dafs
chmod 777 RNAfold
```
Check if ```parse_secondary_structure``` is running, and if not, recompile its cpp file:
```
cd ./GCFM_sources/tools-non-py/lib
g++ parse_secondary_structure.cpp -o parse_secondary_structure
```

## Usage

```
# Simplified tree diagram of files
./GCFM_sources
├── scripts
│   ├── make_profile.py
│   ├── prediction.py
│   ├── splice_profile.py
│   ├── train_cnn_muilt.py
│   ├── train_gc.py
│   ├── utils.py
│   └── WK_NetArch
└── tools-non-py
```
First, use the ```make_profile.py``` to extract the features
```
cd ./GCFM_sources/scripts
python make_profile.py -i "../data/sequence_all.fa" -o "../features/profile/"
```
These features are then used to build a dataset using ```splice_profile.py```
```
python splice_profile.py -d "../features/profile/"
```
Use ```train_cnn_muilt.py``` to train the convolution module of GCFM and turn the features into what is needed for cascade module
```
python train_cnn_muilt.py -d "../features/profile/"
```
Use ```train_gc.py``` to train the cascade module and get the final result
```
python train_gc.py
```
```prediction.py``` is used to apply the model to measure the function of the pairwise sequences relationship
```
python train_gc.py -s1 ">SNORD16_ENST00000362803.1\nTGCAATGATGTCGTAATTTGCGTCTTACTCTGTTCTCAGCGACAGTTGCCTGCTGTCAGTAAGCTGGTACAGAAGGTTGACGAAAATTCTTACTGAGCA" -s2 ">SNORA58_ENST00000505219.1\nGGGCATACTCGTAGACCTTGCCTGACTGTGCTCATGTCCAGGCAGGGGGGACAGTGTATGCAAGAATAATTTGGAGTTCCTGCCAGCTCTAACCAGCTTCATCAGTGGCTGGATAAATTGCAGGACTCTAAACATTT"
```