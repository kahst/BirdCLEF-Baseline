#!/bin/sh
if [ -f /.dockerenv ]; then
  # only run inside docker container
  if [ ! -d "/datasets/train" ]; then
    echo "--- 1/3 sort data ---"
    python ./sort_data.py
  fi

  if [ ! -d "/datasets/spec" ]; then
    echo "--- 2/3 create specs ---"
    python ./spec.py
  fi

  echo "--- 3/3 evaluation ---"
  python evaluate.py
else
  # run outside docker
  if [ "$1" != "" ] && [ -d $1 ]; then
    # (re)build image
    docker build --tag birdclef .
    
    # run with first argument as datasets path
    docker run --mount type=bind,src="$1",dst="/birdclef/datasets" --runtime=nvidia -it --rm birdclef
    
    # delete image (to reduce footprint)
    docker rmi birdclef
  else
    echo "please specify datasets path like this:"
    echo "$0 <datasets directory>"
  fi
fi
