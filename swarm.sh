#!/bin/bash

# Script to run two drivers to test swarm intelligence
./start.sh -p 3001 &
./start.sh -p 3002

function finish {
  pkill -f ./start.sh
}
trap finish EXIT
