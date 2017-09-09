#!/bin/usr/python
# -*- coding: utf-8 -*-

import numpy
import scipy.special
import matplotlib.pyplot

#neural network class definition
class neuralNetwork:

	#initialise the neural network
	def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
		#set number of nodes in each inout, hidden, output layer
		self.inodes=inputnodes
		self.hnodes=hiddennodes
		self.onodes=outputnodes

		#learning rate
		self.lr=learningrate

		#gewichtsmatrizen wih and who
		#weigths im array w_i_j, wo der link von node i zu node j im nächsten layer geht
		#w11 w21 w31 etc
		#w12 w22 w32 etc
		#w13 w23 w33 etc
		#initialisierung mit 1/Wurzel(Anzahl der eingehenden Verknüfpungen), -0.5 um sicher zu stellen
		#das alle zahlen zwischen -1 und 1 sind, 0 darf nicht vorkommen.	
		self.wih=numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
		self.who=numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
		
		#activation funtion is the sigmoid function
		self.activation_function=lambda x: scipy.special.expit(x)
		pass

	#train neural netork
	def train(self, inputs_list, targets_list):
		#convert inputs into 2D array
		inputs = numpy.array(inputs_list, ndmin=2).T
		targets = numpy.array(targets_list, ndmin=2).T
		
		#calculate signals into hidden layer
		hidden_inputs = numpy.dot(self.wih, inputs)
		#calculate the signals emerging from hidden layer
		hidden_outputs = self.activation_function(hidden_inputs)
		
		#calculate signals into final output layer
		final_inputs = numpy.dot(self.who, hidden_outputs)
		#calcualte the signals emerging from final output layer
		final_outputs = self.activation_function(final_inputs)
		
		#error is the (target - actual)
		output_errors = targets - final_outputs
		
		#hidden layer errors is the output_errors, split by weights, recombined at hidden nodes
		hidden_errors = numpy.dot(self.who.T, output_errors)
		
		#update the weights for the links between the hidden and output layers
		self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
		
		#update the weights for the links between the input and hidden layers
		self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
		pass


    #query neural network
	def query(self, inputs_list):
		#convert inputs list to 2d array
		inputs = numpy.array(inputs_list, ndmin=2).T
		
		#calculate signals into hidden layer
		hidden_inputs = numpy.dot(self.wih, inputs)
		
		#calculate the signasl emerging from hidden layout
		hidden_outputs=self.activation_function(hidden_inputs)
		
		#calculate signals into final output layer
		final_inputs=numpy.dot(self.who, hidden_outputs)
		#calculate the signals emerging from final output layer
		final_outputs = self.activation_function(final_inputs)
		return final_outputs
		
def main():
	inputNodes=784
	hiddenNodes=200
	outputNodes=10
	learningRate=0.2
	n=neuralNetwork(inputNodes, hiddenNodes, outputNodes, learningRate)
	
	epochs = 2
	
	#load the mnist training data
	#training_data_file = open("/data/mnist_train_100.csv",'r')
	training_data_file = open("/data/mnist_train.csv",'r')
	training_data_list = training_data_file.readlines()
	training_data_file.close()
	   
	#train the neural network
	for e in range(epochs):
		#go tgrough all records in the training data set
		for record in training_data_list:
			#split record by ',' commas
			all_values = record.split(",")
			#scale and shift the inputs
			inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
			#create the target output values (all 0.01, except the desired label which i 0.99)
			targets = numpy.zeros(outputNodes) + 0.01
			#all_values[0] us the target label for this record
			targets[int(all_values[0])] = 0.99
			n.train(inputs, targets)
			pass
		pass
		
	#load the mnist test data csv file into a list
	#test_data_file = open("/data/mnist_test_10.csv",'r')
	test_data_file = open("/data/mnist_test.csv",'r')
	test_data_list = test_data_file.readlines()
	test_data_file.close()
		
	#test the neural_network
	#scorecard for how well the network performs, initially empty
	scorecard = []
		
	#go through all the records in the test data set
	for record in test_data_list:
		#split the record by the ','
		all_values=record.split(",")
		#correct answer is first values
		correct_label=int(all_values[0])
		print(correct_label, "correct label")
		#scale and shift the inputs
		inputs=(numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
		#query the network
		outputs = n.query(inputs)
		label=numpy.argmax(outputs)
		print(label, "network's label")
		#append correct or incorrect to list
		if (label == correct_label):
			#networks answer matches correct answer, add 0 to scrorecard
			scorecard.append(1)
		else:
			#networks answer doesn't match correct answer, add 0 to scorcard
			scorecard.append(0)
			pass
		pass	
	#calculate the performance score, the fraction of correct answers
	scorecard_array = numpy.asarray(scorecard)
	print("performance = ", scorecard_array.sum() / scorecard_array.size)
			
if __name__ == '__main__':
   main()   
