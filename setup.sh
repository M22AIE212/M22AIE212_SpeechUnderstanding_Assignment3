#!/bin/bash

# Clone the repository
git clone https://github.com/TakHemlata/SSL_Anti-spoofing.git

# Install required Python packages
pip install torch==1.11.0
pip install torchvision==0.12.0
pip install torchaudio==0.11.0

# Change directory to fairseq
cd SSL_Anti-spoofing/fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1

# Install fairseq
pip install --editable .

# Move back to the parent directory
cd ../

# Install requirements
pip install -r requirements.txt

# Install specific numpy version
pip install numpy==1.22.4
