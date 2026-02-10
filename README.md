# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model


<img width="766" height="873" alt="Screenshot 2026-02-09 221457" src="https://github.com/user-attachments/assets/e478b0fd-c9a3-491a-b30f-7554a78ce629" />


## DESIGN STEPS

### STEP 1:
Import the required libraries for data handling and neural networks.

### STEP 2:
Load the dataset and explore its structure.

### STEP 3:
Clean the dataset and handle missing values if present.

### STEP 4:
Encode categorical variables into numerical format.

### STEP 5:
Normalize or scale the numerical features.

### STEP 6:
Split the dataset into training and testing sets.

### STEP 7:
Define the neural network architecture (64 → 32 → 16 → 8 → 4).

### STEP 8:
Select CrossEntropyLoss as the loss function and Adam as the optimizer.

### STEP 9:
Train the model using forward pass, loss calculation, backpropagation, and weight updates.

### STEP 10:
Evaluate the model using accuracy, confusion matrix, and classification report.

## PROGRAM

### Name: SHARAN G 
### Register Number:212223230203

```
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 8)
        self.fc5 = nn.Linear(8, 4)
    def forward(self, x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        x=F.relu(self.fc4(x))
        x=self.fc5(x)
        return x
        


```
```
# Initialize the Model, Loss Function, and Optimizer
model = PeopleClassifier(input_size=X_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.01)

```
```
def train_model(model, train_loader,criterion,optimizer,epochs=100):
  for epoch in range(epochs):
    model.train()
    for X_batch, y_batch in train_loader:
      optimizer.zero_grad()
      output = model(X_batch)
      loss = criterion(output,y_batch)
      loss.backward()
      optimizer.step()

    if (epoch + 1) % 10 == 0:
      print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
        

```



## Dataset Information



<img width="1047" height="782" alt="image" src="https://github.com/user-attachments/assets/23678a59-26d6-4642-8c72-1f07e83c8b1d" />







## OUTPUT



### Confusion Matrix

<img width="824" height="579" alt="Screenshot 2026-02-09 213757" src="https://github.com/user-attachments/assets/1ec2561c-8a2e-49b1-8631-d30451962e27" />


### Classification Report




<img width="623" height="428" alt="Screenshot 2026-02-09 213850" src="https://github.com/user-attachments/assets/46d3f26c-0bea-4bac-b3d5-b1f7e4ca5482" />




### New Sample Data Prediction



<img width="384" height="97" alt="Screenshot 2026-02-09 213923" src="https://github.com/user-attachments/assets/3423a01e-6fc2-4cfa-bc91-360e44ec1fb0" />



## RESULT
Thus a neural network classification model for the given dataset is executed successfully.
