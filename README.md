# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model
<img width="833" height="619" alt="image" src="https://github.com/user-attachments/assets/797a7e7f-4626-4764-8cf2-9268d0af7151" />



Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:
Import necessary libraries and load the dataset.

### STEP 2:
Encode categorical variables and normalize numerical features.

### STEP 3:
Split the dataset into training and testing subsets.
### STEP 4:
Design a multi-layer neural network with appropriate activation functions.
### STEP 5:
Train the model using an optimizer and loss function.
### STEP 6:
Evaluate the model and generate a confusion matrix.
### STEP 7:
Use the trained model to classify new data samples.
### STEP 8:
Display the confusion matrix, classification report, and predictions.


## PROGRAM

### Name: SHARAN G
### Register Number:212223230203

```
class PeopleClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(PeopleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

```
```python
# Initialize the Model, Loss Function, and Optimizer
input_size = X_train.shape[1]
num_classes = 4

model = PeopleClassifier(input_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

```
```python
def train_model(model, train_loader, criterion, optimizer, epochs):
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")
```



## Dataset Information
<img width="847" height="374" alt="image" src="https://github.com/user-attachments/assets/3b614c52-5dee-4ccb-bb71-a0989102a655" />




## OUTPUT
### Confusion Matrix
<img width="805" height="717" alt="image" src="https://github.com/user-attachments/assets/bf52ccfc-37a1-4956-91c7-fb945f87c155" />







### Classification Report
<img width="760" height="302" alt="image" src="https://github.com/user-attachments/assets/6e0f01a6-0de1-4e00-8efe-962882f0260e" />




### New Sample Data Prediction
<img width="785" height="270" alt="image" src="https://github.com/user-attachments/assets/67fd9c7e-69e5-4e64-aea5-0d2ff78c6cef" />



### Result
Thus the neural network classification model was successfully developed.
