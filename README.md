# Deel
Deel; A High level deep neural network description language.

***Now Under construction***

![logo](deel.png)


## Goal
Describe deep neural network, training and using in simple syntax.

###Examples

####CNN classifier
```python
CNN = GoogLeNet()

Input("deel.png")
CNN.classify()
Show()

```

####CNN trainer
```python
CNN = GoogLeNet()

InputStream(train='train.tsv',
			test='test.tsv')
Train(lambda x,t:
	CNN.classify(x)
	return CNN.loss(t)
)
```

####CNN-LSTM trainer
```python
CNN = GoogLeNet()
RNN = LSTM()

InputStream(train='train.tsv',
			test='test.tsv')
Train(lambda x,t:
	Input(x)
	CNN.classify() 
	RNN.forward()
	return CNN.loss(t)
,epoch=500)
```
