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
TrainInput('train.tsv') 
TestInput('test.tsv') 
ValInput('val.tsv') 
CNN.caffenet() 
Train(epoch=500) 
```

####CNN-LSTM trainer
```python
TrainInput('train.tsv') 
TestInput('test.tsv') 
ValInput('val.tsv') 
CNN.caffenet() 
Gap()
LSTM(units=10,num_of_layers=5)
Train() 
```
