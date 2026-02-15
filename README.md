<H3>Name : THIRUMALAI K</H3>
<H3>Register no. : 212224240176</H3>
<H3>Date : 14/02/2026</H3>
<H3>Experiment No. 2 </H3>
## Implementation of Perceptron for Binary Classification
# AIM:
To implement a perceptron for classification using Python<BR>

# EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

# RELATED THEORETICAL CONCEPT:
A Perceptron is a basic learning algorithm invented in 1959 by Frank Rosenblatt. It is meant to mimic the working logic of a biological neuron. The human brain is basically a collection of many interconnected neurons. Each one receives a set of inputs, applies some sort of computation on them and propagates the result to other neurons.<BR>
A Perceptron is an algorithm used for supervised learning of binary classifiers.Given a sample, the neuron classifies it by assigning a weight to its features. To accomplish this a Perceptron undergoes two phases: training and testing. During training phase weights are initialized to an arbitrary value. Perceptron is then asked to evaluate a sample and compare its decision with the actual class of the sample.If the algorithm chose the wrong class weights are adjusted to better match that particular sample. This process is repeated over and over to finely optimize the biases. After that, the algorithm is ready to be tested against a new set of completely unknown samples to evaluate if the trained model is general enough to cope with real-world samples.<BR>
The important Key points to be focused to implement a perceptron:
Models have to be trained with a high number of already classified samples. It is difficult to know a priori this number: a few dozen may be enough in very simple cases while in others thousands or more are needed.
Data is almost never perfect: a preprocessing phase has to take care of missing features, uncorrelated data and, as we are going to see soon, scaling.<BR>
Perceptron requires linearly separable samples to achieve convergence.
The math of Perceptron. <BR>
If we represent samples as vectors of size n, where ‘n’ is the number of its features, a Perceptron can be modeled through the composition of two functions. The first one f(x) maps the input features  ‘x’  vector to a scalar value, shifted by a bias ‘b’
f(x)=w.x+b
 <BR>
A threshold function, usually Heaviside or sign functions, maps the scalar value to a binary output:

 


<img width="283" alt="image" src="https://github.com/Lavanyajoyce/Ex-2--NN/assets/112920679/c6d2bd42-3ec1-42c1-8662-899fa450f483">


Indeed if the neuron output is exactly zero it cannot be assumed that the sample belongs to the first sample since it lies on the boundary between the two classes. Nonetheless for the sake of simplicity,ignore this situation.<BR>


# ALGORITHM:
STEP 1: Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Plot the data to verify the linear separable dataset and consider only two classes<BR>
STEP 4:Convert the data set to scale the data to uniform range by using Feature scaling<BR>
STEP 4:Split the dataset for training and testing<BR>
STEP 5:Define the input vector ‘X’ from the training dataset<BR>
STEP 6:Define the desired output vector ‘Y’ scaled to +1 or -1 for two classes C1 and C2<BR>
STEP 7:Assign Initial Weight vector ‘W’ as 0 as the dimension of ‘X’
STEP 8:Assign the learning rate<BR>
STEP 9:For ‘N ‘ iterations ,do the following:<BR>
        v(i) = w(i)*x(i)<BR>
         
        W (i+i)= W(i) + learning_rate*(y(i)-t(i))*x(i)<BR>
STEP 10:Plot the error for each iteration <BR>
STEP 11:Print the accuracy<BR>
# PROGRAM:
```C
# --------------------------------------------
# IMPORT REQUIRED LIBRARIES
# --------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# --------------------------------------------
# PERCEPTRON CLASS
# --------------------------------------------
class Perceptron:
    def __init__(self, learning_rate=0.1):
        self.lr = learning_rate
        self.w = None
        self.b = 0
        self.errors = []

    # Train the perceptron
    def fit(self, X, y, epochs=10):
        self.w = np.zeros(X.shape[1])
        self.b = 0
        self.errors = []

        for _ in range(epochs):
            error = 0

            # shuffle data for better learning
            indices = np.random.permutation(len(X))
            for i in indices:
                xi = X[i]
                yi = y[i]

                y_pred = self.predict(xi)
                update = self.lr * (yi - y_pred)

                self.w += update * xi
                self.b += update

                if update != 0:
                    error += 1

            self.errors.append(error)

    # Net input
    def net_input(self, x):
        return np.dot(x, self.w) + self.b

    # Predict output
    def predict(self, x):
        return np.where(self.net_input(x) >= 0, 1, -1)


# --------------------------------------------
# READ IRIS DATASET (EXCEL)
# --------------------------------------------
df = pd.read_excel("Iris.xlsx")

print("Dataset Preview:")
print(df.head())


# --------------------------------------------
# 3D PLOT – ALL 3 CLASSES
# --------------------------------------------
X3 = df.iloc[:, 0:3].values
y_all = df.iloc[:, 4].values

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title("Iris Dataset (3D View)")
ax.set_xlabel("Sepal Length")
ax.set_ylabel("Sepal Width")
ax.set_zlabel("Petal Length")

ax.scatter(X3[:50,0], X3[:50,1], X3[:50,2], color='red', label='Setosa')
ax.scatter(X3[50:100,0], X3[50:100,1], X3[50:100,2], color='blue', label='Versicolor')
ax.scatter(X3[100:150,0], X3[100:150,1], X3[100:150,2], color='green', label='Virginica')

ax.legend()
plt.show()


# --------------------------------------------
# BINARY CLASSIFICATION (SETOSA vs VERSICOLOR)
# --------------------------------------------
X = df.iloc[:100, 0:2].values   # two features
y = df.iloc[:100, 4].values

# 2D Plot
plt.scatter(X[:50,0], X[:50,1], color='red', marker='o', label='Setosa')
plt.scatter(X[50:100,0], X[50:100,1], color='blue', marker='x', label='Versicolor')
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.legend()
plt.show()


# --------------------------------------------
# LABEL ENCODING
# --------------------------------------------
y = np.where(y == 'Iris-setosa', 1, -1)


# --------------------------------------------
# FEATURE STANDARDIZATION
# --------------------------------------------
X[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()


# --------------------------------------------
# TRAIN-TEST SPLIT
# --------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)


# --------------------------------------------
# TRAIN PERCEPTRON
# --------------------------------------------
model = Perceptron(learning_rate=0.1)
model.fit(X_train, y_train, epochs=10)


# --------------------------------------------
# ACCURACY
# --------------------------------------------
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred) * 100)


# --------------------------------------------
# ERROR vs EPOCH GRAPH
# --------------------------------------------
plt.plot(range(1, len(model.errors)+1), model.errors, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Errors")
plt.title("Perceptron Training Errors")
plt.show()

```

# OUTPUT:

<img width="758" height="740" alt="image" src="https://github.com/user-attachments/assets/0b45d068-12be-46b2-a3a2-110f0a7ab0dd" />
<img width="632" height="538" alt="image" src="https://github.com/user-attachments/assets/95d32cf6-bfc4-4b79-987f-dccd46e5db07" />
<img width="623" height="502" alt="image" src="https://github.com/user-attachments/assets/0490972b-96a3-4fb5-806d-29797596692e" />




# RESULT:
 Thus, a single layer perceptron model is implemented using python to classify Iris data set.

 
