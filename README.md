# Deel
Deel; A High level deep neural network description language.

***Under construction***

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
Input('dataset.py') 
CNN.caffenet() 
Train() 
```

####CNN-LSTM trainer
```python
Input('dataset.py') 
CNN.caffenet() 
Gap()
LSTM(units=10,num_of_layers=5)
Train() 
```
