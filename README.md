<h1>Assignment-mid-term</h1>  

Build a neural network using python **without any machine-learning platform.**
  
_This project can be divided to 4 parts_
  
---  

1. Build Neural network with out `tensor-flow`.  
(In my case, just used `scikit-learn` for load data.)

2. Test the Network by **changing the activation function**.  
(In this project, `sigmoid`, `Relu` are included.)

3. Test the Network by **changing the optimizer**.  
(This project contains `Gradient-Descent`, `AdaGrad` and `Adam`-optimizer.)

4. Test the Network by **changing the batch size**.  
(I tested batch, mini-batch and stochastic batch size.) 

  
---

<h3>Structure of project</h3>

~~~
-assignment-mid-term : This project is my assignment.
  ├─ -nn : packge of my own network.
  │   │
  │   ├─ -datas.py : Data-Manager for this model. Load data, divide data, etc..
  │   │
  │   ├─ -model.py : Main network models include activate functions, feed-forword, bac
  │   │
  │   └─ -tools.py : Plotting tools.
  │
  ├─ -assignment.py : main stream of this project.
  │
  └─ -ema_test.py : Before constructing the Adam optimizer, i need to understand how calculat the Expotential moving average.
                    This is the test of caculating the ema. 
~~~

 
---
 
<h3>structure of `assignment.py`</h3>

First, setting the params for data using config dic data.   
This config has many params and you can change the `epoch`, `learning_rate`, `batch_size`, `activation`, `optimizer`, etc...  
~~~
config_assignmentA = {'total_epoch': 120, 'batch_size': 1000, 'learning_rate': 0.1, 'random_seed': 42,
                      'train_dataset_size': 60000, 'test_dataset_size': 10000, 'optimizer': nn.model.OPTIMIZER_GD,
                      'activation': nn.model.ACTIVATE_SIGMOID}
~~~  
<br/>

Next, define network_model and load dataManager.
~~~
# define network nn.
network_model = network(configure=config_assignmentD_ADAM, h1=100, h2=50)
dataManager = data_manager()
~~~

<br/>

Training network is simple. just load the batch data, and run **train()** function.  
Train method _update all weights one time_ because understand how can back-porpagation work in network. So just use kind of `for loop` for train the network.
~~~
# load batch data.
batch_x, batch_y = dataManager.next_batch(network_model.BATCH_SIZE)

# train model.
network_model.train(batch_x, batch_y)
~~~
  
  
<br/>

Calculate Accuracy and loss by using network function.  
See below code.

~~~
# calculate accuracy and loss
output_train = network_model.predict(dataManager.X_train)
accuracy_train, loss_train = network_model.getAccuracyAndLoss(output_train, dataManager.y_train)
~~~


 