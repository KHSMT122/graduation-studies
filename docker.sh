#!/bin/bash

CONTAINER_NAME=hoge1
IMAGES=keita
TAGS=step2
PORT=1111

docker run --rm -it --gpus all --ipc host -v $PWD:$PWD -v ~/.dockerssh:/root/.ssh:ro -p ${PORT}:${PORT} --name ${CONTAINER_NAME} ${IMAGES}:${TAGS}

#run "umask 000" after this script
#docker makes file or dir by root, therefore make permission 777
