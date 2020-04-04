import numpy as np

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
  fx = sigmoid(x)
  return fx * (1 - fx)

def mse_loss(y_true, y_pred):
  return ((y_true - y_pred) ** 2).mean()

class NN:
    def __init__(self,n_i,n_h,n_o):
        self.numberofinput=n_i
        self.numberofhidden=n_h
        self.numberofoutput=n_o
        # Weights of input to hidden layer shape is (nh,ni)
        self.w_i_h=np.random.random((self.numberofhidden,self.numberofinput))
        # Weights of hidden to output layer
        self.w_h_o=np.random.random((self.numberofoutput,self.numberofhidden))
        # Bias of hidden layer,shape is(nh,1)
        self.b_h=np.random.random((self.numberofhidden,1))
        # Bias of output layer
        self.b_o=np.random.random((self.numberofoutput,1))
        # Hidden and output layer neurons
        self.h=np.zeros((self.numberofhidden,1))
        self.output=np.zeros((self.numberofoutput,1))
    def feedforward(self,x):
        # Feed Forward the network
        x=x.reshape((self.numberofinput,1))
        self.h=sigmoid(self.w_i_h.dot(x)+self.b_h)
        self.output=sigmoid(self.w_h_o.dot(self.h)+self.b_o)
        return self.output
    def train(self,inputdata,actual_output,learningrate=0.10,epochs=10000):
        for epoch in range(epochs):
            for x,y_actual in zip(inputdata,actual_output):
                y_predict=self.feedforward(x)
                # Partial Derivative of loss function with respect to predicted value
                dl_dy=-2*(y_actual-y_predict)
                
                # Partial Derivative predictted value with respect to hidden-output weights and output bias
                dy_dwho=self.h*deriv_sigmoid(self.w_h_o.dot(self.h)+self.b_o)
                dy_dbo=deriv_sigmoid(self.w_h_o.dot(self.h)+self.b_o)
                
                # Partial Derivative of predicted value with respect to hidden layer
                dy_dh=self.w_h_o*deriv_sigmoid(self.w_h_o.dot(self.h)+self.b_o)
                
                # Partial Derivative of predicted value with respect to input-hidden weights and hidden bias
                dh_dwih=x*deriv_sigmoid(self.w_i_h.dot(x)+self.b_h)
                dh_dbh=deriv_sigmoid(self.w_i_h.dot(x.reshape(self.numberofinput,1))+self.b_h)

                # Updating values
                self.w_i_h-=learningrate*dl_dy*dy_dh*dh_dwih
                self.b_h  -=learningrate*dl_dy*dh_dbh
                self.w_h_o-=learningrate*dl_dy*dy_dwho.reshape(1,self.numberofhidden)
                self.b_o  -=learningrate*dl_dy*dy_dbo
                
# Inputs for neural network
data = np.array([[0,0],[0,1],[1,0],[1,1],])

# Output for different Function
orFunction = np.array([0,1,1,1,])
andFunction = np.array([0,0,0,1,])


# Create Trainer for function
orTrainer=NN(2,2,1)
andTrainer=NN(2,2,1)


trainers=[orTrainer,andTrainer]
FunctionActualOutputs=[orFunction,andFunction]

for f,actual_output in zip(trainers,FunctionActualOutputs):
    f.train(data,actual_output)
    for i in range(len(data)):
        predicted=float(f.feedforward(data[i]))
        print(f"Input {data[i]} Actual value {actual_output[i]} Predicted Value {predicted} ")
    print("\n")
