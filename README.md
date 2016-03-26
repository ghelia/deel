# Deel
Deel; A High level deep neural network description language.

***Now Under construction***

![logo](deel.png)


## Goal
Describe deep neural network, training and using in simple syntax.

## Install and test

$ git clone https://github.com/uei/deel.git
$ cd deel/data
$ ./getCaltech101.sh
$ cd ..
$ python test.py

###Examples

####CNN classifier (done)
```python
CNN = GoogLeNet()

Input("deel.png")
CNN.classify()
Show()

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

####CNN-LSTM trainer (not yet)
```python
CNN = GoogLeNet()
RNN = LSTM()

InputBatch(train='train.tsv',
			test='test.tsv')
def workout(x,t):
	Input(x)
	CNN.classify() 
	RNN.forward()
	return RNN.backprop(t)

BatchTrain(epoch=500)
```

####CNN-DQN with Unity (not yet)
```python
CNN = GoogLeNet()
DQN = DeepQLearning(output=4)

def workout():
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

