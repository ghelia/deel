# Deel
Deel; A High level deep neural network description language.

***Now Under construction***

![logo](deel.png)


## Goal
Describe deep neural network, training and using in simple syntax.

## Install and test

```sh
$ git clone https://github.com/uei/deel.git
$ cd deel/data
$ ./getCaltech101.sh
$ cd ../misc
$ ./getCaffeNet.sh
$ cd ..
$ python test.py
```

###Examples

####CNN classifier (done)
```python
deel = Deel()

CNN = GoogLeNet()

CNN.Input("deel.png")
CNN.classify()
ShowLabels()

```

####CNN trainer (almost done,not testing with GPU)
```python
nin = NetworkInNetwork()

InputBatch(train="data/train.txt",
			val="data/test.txt")

def workout(x,t):
	nin.classify(x)	
	return nin.backprop(t)

BatchTrain(workout)
```

####CNN-LSTM trainer (done, not test)
```python
InputBatch(train="data/train_lstm.tsv")

CNN = GoogLeNet()
RNN = LSTM()

def trainer(x,t):
	CNN.classify(x) 
	RNN.learn(t)
	return RNN.backprop()

BatchTrain(trainer)
```

####CNN-DQN with Unity (not yet)
```python
CNN = GoogLeNet()
DQN = DeepQLearning(output=4)

def trainer():
	#Realtime input image from Unity
	InputUnity('unity.png') 
	CNN.classify() 
	DQN.forward()
    OutputUnity( { 0:'left, 1:'right, 2:'up', 3:'down', 4:'space'})

	#Get score or loss from Unity game
	t = InputVarsFromUnity()
	DQN.reinforcement(t)

StreamTrain(workout)
```

