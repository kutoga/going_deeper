#!/bin/bash
docker_user_host=$1
docker_socket=/tmp/docker.sock

if [ -f "$docker_socket" ]; then
    rm "$docker_socket"
fi

echo "Execute export DOCKER_HOST=unix://$docker_socket"

ssh -nNT -L $docker_socket:/var/run/docker.sock $docker_user_host
