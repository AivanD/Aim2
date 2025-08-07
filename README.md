# AIM 2

# Installation instructions
```bash 
pip install -e .
pip install -r requirements.txt

# for models used 
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz # for non-transformer model 
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_scibert-0.5.4.tar.gz # for transformer model which requires pytorch cuda

```