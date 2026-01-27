# Security ML

Paper in question: https://people.csail.mit.edu/devadas/pubs/Learnable_Obfuscation.pdf

### Personal Setup (conda)

```
conda create -n security python=3.11 # rename security to whatever you want
conda activate security
```

### Installing Torch w/ cuda

To utilize cuda, install torch according to this page: https://pytorch.org/get-started/locally/

For example:

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

If no GPU, pip install torch and torchvision normally. 

### Install remaining dependencies

```
pip install -r requirements.txt
```

### Run Code

```
python main.py
```