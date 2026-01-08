#!/bin/bash
#v0.3.0-21nov25
echo -e "\nStarting the 'conda' environment in the 'pesq' area...\n"
/pesq/dados/monan/cursowmo2025/anaconda3/bin/conda init
source ~/.bashrc
cd /mnt/beegfs/$USER
pwd
echo -e "\nConfiguring conda environment 'cursowmoenv2026' in the 'pesq' area...\n"
conda config --add envs_dirs /pesq/share/monan/curso_OMM_INPE_2025/.conda/envs/
conda activate cursowmoenv2025
echo -e "\nActivating Jupyter-Lab on the corresponding aluno-port...\n"
jupyter-lab --no-browser --port $UID --ServerApp.port_retries=0
