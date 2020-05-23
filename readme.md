# GCFM

A deep fusion model with the input of alignment-based feature representations, for the classification of alignments of pairwise sequences of ncRNAs. This model integrates the convolution module with the cascade forest of GcForest, which can learn abstraction features at high levels, adjust part of model architecture automatically during training, and achieve great performance.

## Install

```
# Simplified tree diagram of files
./GCFM
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
1. Install Python packages according to the ```requirements.txt```
2. Check the permissions of ```dafs``` and ```RNAfold``` in directory tools-non-py (need to be executable)
3. Check if ```parse_secondary_structure``` is running, and if not, recompile its cpp file


## Usage

```
# Simplified tree diagram of files
./GCFM
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
1. First, use the ```make_profile.py``` to extract the features
2. These features are then used to build a dataset using ```splice_profile.py```
3. Use ```train_cnn_muilt.py``` to train the convoluation module of GCFM and turn the features into what is needed for cascade module
4. Use ```train_gc.py``` to train the cascade module and get the final result
5. ```prediction.py``` is used to apply the model to measure the function of the pairwise sequences relationship

