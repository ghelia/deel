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
Output()

```

####CNN trainer
```python
CNN = GoogLeNet()

TrainInput('train.tsv') 
TestInput('test.tsv') 
ValInput('val.tsv') 
CNN.train(epoch=500) 
```

####CNN-LSTM trainer
```python
TrainInput('train.tsv') 
TestInput('test.tsv') 
ValInput('val.tsv') 
CNN.classify() 
Gap()
LSTM.train(units=10,num_of_layers=5)
```
