# python-ai
neural network in python

Build the docker file: docker build -t python-ai -f Dockerfile.in  .

create data directory in the container

wget the data you want to use to the data directory

correct test and train data filenames in python script. Adjust learn rate

copy the python script to the container: docker cp neuralNetwork.py f5e8bcb6a162:/data/neuralNetwork.py

run : python neuralNetwork.py
