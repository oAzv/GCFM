# GCFM

A deep fusion model with the input of alignment-based feature representations, for the classification of alignments of pairwise sequences of ncRNAs. This model integrates the convolution module with the cascade forest of GcForest, which can learn abstraction features at high levels, adjust part of model architecture automatically during training, and achieve great performance.

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
Use ```train_cnn_muilt.py``` to train the convoluation module of GCFM and turn the features into what is needed for cascade module
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