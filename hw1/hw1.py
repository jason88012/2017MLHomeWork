import pandas as pd
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import math
import random
import sys
import os

#Test 123123123

# Constants
CONST_TIMES = 9
CONST_ATTRS = 18

# Learning variable
LEARNING_RATE = np.array([0.03]*(CONST_TIMES*CONST_ATTRS+1))
LAMBDA = 0.1
BATCH_SIZE = 240
EPISODES = 1000

# Define saving path
SAVE_PATH = './hw1_weight.pkl'
RESULT_PATH = './hw1_result.csv'

def dataPreProcess(train, test):
	# Read and process training data, make it's formal as same as testing one
	training_df = pd.read_csv(train, na_values=['.'], header=None, encoding='big5')
	training_df.replace('NR', 0, inplace=True)
	training_df.drop(training_df.index[0], inplace=True)
	training_df.drop(training_df.columns[1], axis=1, inplace=True)
	training_df.reset_index(drop=True, inplace=True)
	training_df.columns = [i for i in range(training_df.shape[1])]
	# Read and process testing data
	test_df = pd.read_csv(test, na_values=['.'], header=None, encoding='big5')
	test_df.replace('NR', 0, inplace=True)
	return training_df, test_df

class record(object):
	def __init__(self):
		self.memory_size = 200
		self.weight = np.ones(CONST_TIMES*CONST_ATTRS + 1)
		self.episodes = 0
		#self.grad_sum = np.zeros(CONST_TIMES*CONST_ATTRS + 1)
		self.grad_sum = np.array([0.0001] * (CONST_TIMES*CONST_ATTRS + 1))
		self.loss_history = []


class Learning(object):
	def __init__(self, datas, tdatas, lr, lamb, batch):
		if os.path.isfile(SAVE_PATH):
			f = open(SAVE_PATH, 'rb')
			self.record = pkl.load(f)
		else:
			self.record = record()
		self.datas = datas
		self.test_datas = tdatas
		self.data_num = int((self.datas.shape[0]-1) / CONST_ATTRS)
		self.learning_rate = lr
		self.lamb = lamb
		self.batch = batch

	def getFeature(self, id, datas):
		feature = []
		for time in range(CONST_TIMES):
			for attr in range(CONST_ATTRS):
				feature.append(float(datas[time+2][id*CONST_ATTRS+attr]))
		# Add bias feature = 1
		feature.append(1)
		return np.array(feature)

	def getRealOutput(self, id, datas):
		return float(datas[CONST_TIMES+2][id*CONST_ATTRS+10])

	# calculate loss fuction
	def loss(self, start, data):
		id = start
		loss = 0
		for _ in range(self.batch):
			if id >= self.batch:
				id = 0
			y = self.getRealOutput(id, data)
			x = self.getFeature(id, data)
			loss += (y - np.dot(self.record.weight, x.T))**2
		return loss / self.batch

	# calculate gradient and normalize it
	def calcGradient(self, start):
		id = start
		grad = np.zeros(CONST_TIMES*CONST_ATTRS + 1)
		for _ in range(self.batch):
			if id >= self.data_num:
				id = 0
			y = self.getRealOutput(id, self.datas)
			x = self.getFeature(id, self.datas)
			g = 2*(y - np.dot(self.record.weight, x.T))*(-x)
			norm = np.linalg.norm(g)
			grad += g/norm
		return grad

	def update(self, start):
		grad = self.calcGradient(start)
		self.record.grad_sum += grad**2
		lr_ada = self.learning_rate / (self.record.grad_sum**0.5)
		self.record.weight = self.record.weight - lr_ada*grad

	def training(self):
		training_times = 0
		for ep in range(EPISODES):
			try:
				start = random.randint(0, self.data_num)
				loss = self.loss(start, self.datas)
				print("Episode:", self.record.episodes, "loss:", loss)
				self.update(start)
				# Record history
				if len(self.record.loss_history) == self.record.memory_size:
					del self.record.loss_history[0]
				self.record.loss_history.append(loss)
				self.record.episodes += 1
			except KeyboardInterrupt:
				break

	def save(self, path):
		if os.path.isfile(path):
			os.remove(path)
		f = open(path, 'wb')
		pkl.dump(self.record, f)
		f.close()

	def test(self):
		fout = open(RESULT_PATH, 'w')
		fout.write('id,value\n')
		for id in range(int(self.test_datas.shape[0] / CONST_ATTRS)):
			id_name = self.test_datas[0][id*CONST_ATTRS]
			x = self.getFeature(id, self.test_datas)
			result = np.dot(self.record.weight, x.T)
			fout.write(id_name+','+str(result)+'\n')
		self.plot(self.record.episodes, self.record.loss_history)

	def plot(self, ep, loss):
		if ep <= self.record.memory_size - 1:
			x = range(ep)
		else:
			x = range(ep-self.record.memory_size, ep)
		y = loss
		avg_loss = sum(self.record.loss_history) / len(self.record.loss_history)
		axis = plt.gca()
		axis.set_ylim([0, avg_loss*1.5])
		plt.plot(x, y)
		plt.title('9 hour attributes training process')
		plt.xlabel('episodes')
		plt.ylabel('loss')
		plt.show()

if __name__ == '__main__':
	train_datas, test_datas = dataPreProcess(sys.argv[1], sys.argv[2])
	l = Learning(train_datas, test_datas, lr=LEARNING_RATE, lamb=LAMBDA, batch=BATCH_SIZE)
	l.training()
	l.save(SAVE_PATH)
	l.test()

