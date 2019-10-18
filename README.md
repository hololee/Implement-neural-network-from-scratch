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
 
<h3>structure of model.py</h3>
In the `model.py` scripts, many functions for training are located.  
-----------------------
1. Initialize model
2. Sigmoid activation model
3. ReLU activation model
4. Feedforward & backpropagation calculate
5. Update calculate (gradient-descent, Adagrad, Adam)
-----------------------


 - Initialize model 
 First, setting network by the user configuration.  
 If use the `Adagrad` or `Adam` optimizer, initialize additional params.
 ~~~
# weight initialize
self.w1 = np.random.randn(784, h1) / 10
self.w2 = np.random.randn(h1, h2) / 10
self.w3 = np.random.randn(h2, 10) / 10

# set configure.
self.configure = configure

# config data.
self.TOTAL_EPOCH = configure['total_epoch']
self.BATCH_SIZE = configure['batch_size']
self.LEARNING_RATE = configure['learning_rate']
self.SEED = configure['random_seed']
self.OPTIMIZER = configure['optimizer']
self.ACTIVATION = configure['activation']

if self.OPTIMIZER == OPTIMIZER_ADAGRAD:
    self.gt_w1 = np.zeros(self.w1.shape)
    self.gt_w2 = np.zeros(self.w2.shape)
    self.gt_w3 = np.zeros(self.w3.shape)
    self.eps = configure['epsilon']

if self.OPTIMIZER == OPTIMIZER_ADAM:
    self.beta1 = configure['beta1']
    self.beta2 = configure['beta2']
    self.eps = configure['epsilon']

    # for calculate beta.
    self.counts = 1

    self.mt_w1 = np.zeros(self.w1.shape)
    self.vt_w1 = np.zeros(self.w1.shape)

    self.mt_w2 = np.zeros(self.w2.shape)
    self.vt_w2 = np.zeros(self.w2.shape)

    self.mt_w3 = np.zeros(self.w3.shape)
    self.vt_w3 = np.zeros(self.w3.shape)
 ~~~  
 
 
 - Sigmoid activation model  
 This is for calculate `sigmoid` feedfoward and derivative of sigmoid.
 ~~~
def sigmoid(self, x):
    return 1.0 / (1.0 + np.exp(-x))

def back_sigmoid(self, x):
    return x * (1. - x)
 ~~~  
 
  - ReLU activation model  
 This is for calculate `relu` feedfoward and derivative of sigmoid.
 ~~~
# included back propagation.
def relu(self, x):
    back_relu = np.zeros(x.shape)
    back_relu[np.where(x > 0)] = 1
    x[np.where(x <= 0)] = 0
    
    return x, back_relu
 ~~~
 
   - Feedforward & backpropagation calculate  
 Every one iteration do one feedforward and one backpropagation.  
 After that **update the weights**.   
 Activate function affect to caculate the feedforward and backpropagation, so divide the source by the activate type.
 ~~~.
def feedForward(self, x):
    y1 = np.dot(x, self.w1)
    if self.ACTIVATION == ACTIVATE_SIGMOID:
        activated_y1 = self.sigmoid(y1)
        back_relu_w1 = None
    elif self.ACTIVATION == ACTIVATE_RELU:
        activated_y1, back_relu_w1 = self.relu(y1)
    else:
        activated_y1 = self.sigmoid(y1)
        back_relu_w1 = None

    y2 = np.dot(activated_y1, self.w2)
    if self.ACTIVATION == ACTIVATE_SIGMOID:
        activated_y2 = self.sigmoid(y2)
        back_relu_w2 = None
    elif self.ACTIVATION == ACTIVATE_RELU:
        activated_y2, back_relu_w2 = self.relu(y2)
    else:
        activated_y2 = self.sigmoid(y2)
        back_relu_w2 = None

    y3 = np.dot(activated_y2, self.w3)
    softmax_result = self.softmax(y3)

    return activated_y1, activated_y2, softmax_result, back_relu_w1, back_relu_w2

def backpropagation(self, x, labelY, out1, out2, out3, back_relu_w1, back_relu_w2):
    d_e = (out3 - labelY) / self.BATCH_SIZE

    # calculate d_w3
    d_w3 = out2.T.dot(d_e)

    # calculate d_w2
    if self.ACTIVATION == ACTIVATE_SIGMOID:
        d_w2 = out1.T.dot(np.matmul(d_e, self.w3.T) * self.back_sigmoid(out2))
    elif self.ACTIVATION == ACTIVATE_RELU:
        d_w2 = out1.T.dot(np.matmul(d_e, self.w3.T) * back_relu_w2)
    else:
        d_w2 = out1.T.dot(np.matmul(d_e, self.w3.T) * self.back_sigmoid(out2))

    # calculate d_w1
    if self.ACTIVATION == ACTIVATE_SIGMOID:
        d_w1 = x.T.dot(
            np.matmul(np.matmul(d_e, self.w3.T) * self.back_sigmoid(out2), self.w2.T) * self.back_sigmoid(out1))
    elif self.ACTIVATION == ACTIVATE_RELU:
        d_w1 = x.T.dot(np.matmul(np.matmul(d_e, self.w3.T) * back_relu_w2, self.w2.T) * back_relu_w1)
    else:
        d_w1 = x.T.dot(
            np.matmul(np.matmul(d_e, self.w3.T) * self.back_sigmoid(out2), self.w2.T) * self.back_sigmoid(out1))

    # return changed value.
    return d_w1, d_w2, d_w3
 ~~~
 
   - Update calculate (gradient-descent, Adagrad, Adam)  
  Calculate weights update.
  
  _If the optimizer is Adagrad, calculate the additional params gt for all weights.  
  If the optimizer is Adam, calculate the additional params mt, vt for all weights._  
  
  
  ![Result of assigmentA](https://github.com/hololee/assignment-mid-term/blob/master/images/algorithm-01.png?raw=true)  
  
  
 ~~~
def update_weight(self, d_w1, d_w2, d_w3):
    if self.OPTIMIZER == OPTIMIZER_GD:
        self.w1 -= self.LEARNING_RATE * d_w1
        self.w2 -= self.LEARNING_RATE * d_w2
        self.w3 -= self.LEARNING_RATE * d_w3

    elif self.OPTIMIZER == OPTIMIZER_ADAGRAD:
        # update the gt.
        self.gt_w1 += np.square(d_w1 ** 2)
        self.gt_w2 += np.square(d_w2 ** 2)
        self.gt_w3 += np.square(d_w3 ** 2)

        # change the learning rate for each weight.
        self.w1 -= (self.LEARNING_RATE / np.sqrt(self.gt_w1 + self.eps)) * d_w1
        self.w2 -= (self.LEARNING_RATE / np.sqrt(self.gt_w2 + self.eps)) * d_w2
        self.w3 -= (self.LEARNING_RATE / np.sqrt(self.gt_w3 + self.eps)) * d_w3

    elif self.OPTIMIZER == OPTIMIZER_ADAM:

        self.mt_w1 = (self.beta1 * self.mt_w1) + ((1 - self.beta1) * d_w1)
        self.vt_w1 = (self.beta2 * self.vt_w1) + ((1 - self.beta2) * (d_w1 ** 2))

        self.mt_w1 = self.mt_w1 / (1 - self.beta1)
        self.vt_w1 = self.vt_w1 / (1 - self.beta2)

        self.mt_w2 = (self.beta1 * self.mt_w2) + ((1 - self.beta1) * d_w2)
        self.vt_w2 = (self.beta2 * self.vt_w2) + ((1 - self.beta2) * (d_w2 ** 2))

        self.mt_w2 = self.mt_w2 / (1 - self.beta1)
        self.vt_w2 = self.vt_w2 / (1 - self.beta2)

        self.mt_w3 = (self.beta1 * self.mt_w3) + ((1 - self.beta1) * d_w3)
        self.vt_w3 = (self.beta2 * self.vt_w3) + ((1 - self.beta2) * (d_w3 ** 2))

        self.mt_w3 = self.mt_w3 / (1 - self.beta1)
        self.vt_w3 = self.vt_w3 / (1 - self.beta2)

        self.counts += 1
        self.beta1 = 2 / (self.counts + 1)
        self.beta2 = 2 / (self.counts + 1)

        self.w1 -= (self.LEARNING_RATE / np.sqrt(self.vt_w1 + self.eps)) * self.mt_w1
        self.w2 -= (self.LEARNING_RATE / np.sqrt(self.vt_w2 + self.eps)) * self.mt_w2
        self.w3 -= (self.LEARNING_RATE / np.sqrt(self.vt_w3 + self.eps)) * self.mt_w3
 ~~~

 
---
 
<h3>structure of assignment.py</h3>

First, setting the params for data using config dic data.   
This config has many params and you can change the `epoch`, `learning_rate`, `batch_size`, `activation`, `optimizer`, etc...  
~~~
config_assignment = {'total_epoch': 120, 'batch_size': 1000, 'learning_rate': 0.1, 'random_seed': 42,
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
![Result of assigmentA](https://github.com/hololee/assignment-mid-term/blob/master/images/network-01.png?raw=true)  

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

 
---
 
<h3>result of model training </h3>

- defined model. (100 h1 layers, 50 h2 layers.)
~~~
# define network nn.
network_model = network(configure=config_assignmentB, h1=100, h2=50)
dataManager = data_manager()
~~~


- network_assignment_A(mini-batch, sigmoid activation)
~~~
config_assignmentA = {'total_epoch': 50, 'batch_size': 1000, 'learning_rate': 0.1, 'random_seed': 42,
                      'train_dataset_size': 60000, 'test_dataset_size': 10000, 'optimizer': nn.model.OPTIMIZER_GD,
                      'activation': nn.model.ACTIVATE_SIGMOID}
                      
                   
       
result.   
--------------------------------------------
-------------- batch 56 training...
-------------- batch 57 training...
-------------- batch 58 training...
-------------- batch 59 training...
============== EPOCH 50 END ================
train accuracy : 0.9062; loss : 0.00738, test accuracy : 0.908; loss : 0.00719                                         
~~~
![Result of assigmentA](https://github.com/hololee/assignment-mid-term/blob/master/images/plot_a.png?raw=true)  


- network_assignment_B(mini-batch, Relu activation)
~~~
config_assignmentB = {'total_epoch': 50, 'batch_size': 1000, 'learning_rate': 0.1, 'random_seed': 42,
                      'train_dataset_size': 60000, 'test_dataset_size': 10000, 'optimizer': nn.model.OPTIMIZER_GD,
                      'activation': nn.model.ACTIVATE_RELU}
                      
                   
       
result.   
--------------------------------------------
-------------- batch 56 training...
-------------- batch 57 training...
-------------- batch 58 training...
-------------- batch 59 training...
============== EPOCH 50 END ================
train accuracy : 0.973; loss : 0.00211, test accuracy : 0.963; loss : 0.00275
~~~
![Result of assigmentB](https://github.com/hololee/assignment-mid-term/blob/master/images/plot_b.png?raw=true)  


- network_assignment_C_1(mini-batch, Relu activation)
~~~
config_assignmentC_MINI_BATCH = {'total_epoch': 50, 'batch_size': 1000, 'learning_rate': 0.1, 'random_seed': 42,
                                 'train_dataset_size': 60000, 'test_dataset_size': 10000,
                                 'optimizer': nn.model.OPTIMIZER_GD,
                                 'activation': nn.model.ACTIVATE_RELU}            
                   
    
       
result.   
--------------------------------------------
-------------- batch 56 training...
-------------- batch 57 training...
-------------- batch 58 training...
-------------- batch 59 training...
============== EPOCH 50 END ================
train accuracy : 0.973; loss : 0.00212, test accuracy : 0.964; loss : 0.0027
~~~
![Result of assigmentC_1](https://github.com/hololee/assignment-mid-term/blob/master/images/plot_c_1.png?raw=true)

- network_assignment_C_2(batch, Relu activation)
~~~
config_assignmentC_BATCH = {'total_epoch': 80, 'batch_size': 60000, 'learning_rate': 0.1, 'random_seed': 42,
                            'train_dataset_size': 60000, 'test_dataset_size': 10000,
                            'optimizer': nn.model.OPTIMIZER_GD,
                            'activation': nn.model.ACTIVATE_RELU}            
                   
    
       
result.   
--------------------------------------------
============== EPOCH 80 START ==============
-------------- batch 0 training...
============== EPOCH 80 END ================
train accuracy : 0.843; loss : 0.0129, test accuracy : 0.849; loss : 0.0125
~~~
![Result of assigmentC_2](https://github.com/hololee/assignment-mid-term/blob/master/images/plot_c_2.png?raw=true)

- network_assignment_C_3(stochastic batch, Relu activation): **Too long time spend for train!!**
~~~
config_assignmentC_STOCHASTIC = {'total_epoch': 30, 'batch_size': 1, 'learning_rate': 0.01, 'random_seed': 42,
                                 'train_dataset_size': 60000, 'test_dataset_size': 10000,
                                 'optimizer': nn.model.OPTIMIZER_GD,
                                 'activation': nn.model.ACTIVATE_RELU}           
                   
    
       
result.   
--------------------------------------------
============== EPOCH 30 START ==============
============== EPOCH 30 END ================
train accuracy : 0.9895; loss : 0.000797, test accuracy : 0.974; loss : 0.00212
~~~
![Result of assigmentC_3](https://github.com/hololee/assignment-mid-term/blob/master/images/plot_c_3.png?raw=true)

- network_assignment_D_1(mini batch, Relu activation, Adagrad)
~~~
config_assignmentD_ADAGRAD = {'total_epoch': 30, 'batch_size': 1000, 'learning_rate': 0.005, 'random_seed': 42,
                              'train_dataset_size': 60000, 'test_dataset_size': 10000,
                              'optimizer': nn.model.OPTIMIZER_ADAGRAD,
                              'activation': nn.model.ACTIVATE_RELU,
                              'epsilon': 1e-5}   
                   
    
       
result.   
--------------------------------------------
-------------- batch 56 training...
-------------- batch 57 training...
-------------- batch 58 training...
-------------- batch 59 training...
============== EPOCH 30 END ================
train accuracy : 0.9968; loss : 0.000328, test accuracy : 0.978; loss : 0.00168
~~~
![Result of assigmentD_1](https://github.com/hololee/assignment-mid-term/blob/master/images/plot_d_1.png?raw=true)


- network_assignment_D_2(mini batch, Relu activation, Adam)
~~~
config_assignmentD_ADAM = {'total_epoch': 30, 'batch_size': 1000, 'learning_rate': 0.005, 'random_seed': 42,
                           'train_dataset_size': 60000, 'test_dataset_size': 10000,
                           'optimizer': nn.model.OPTIMIZER_ADAM,
                           'activation': nn.model.ACTIVATE_RELU,
                           'beta1': 0.9,
                           'beta2': 0.999,
                           'epsilon': 1e-8}  
                   
    
       
result.   
--------------------------------------------
-------------- batch 56 training...
-------------- batch 57 training...
-------------- batch 58 training...
-------------- batch 59 training...
============== EPOCH 30 END ================
train accuracy : 0.9946; loss : 0.000423, test accuracy : 0.974; loss : 0.0023

~~~
![Result of assigmentD_2](https://github.com/hololee/assignment-mid-term/blob/master/images/plot_d_2.png?raw=true)

 