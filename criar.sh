#!/bin/bash

if [ -f environment.yml ]
then
	echo "Informe o nome do ambiente a ser criado a partir do environment.yml (original: bam, sugestão: escolher outro nome!!)"
	read NOME_AMBIENTE
	conda env create -f environment.yml --name $NOME_AMBIENTE
else
	echo "Arquivo environment.yml inexiste!"
fi
