import torch
import torch.nn as nn
from model import PneumoniaCNN
from prep_data import train_loader, test_loader, val_loader

# create the model , loss function, and optimizer - the core pieces needed for training
model = PneumoniaCNN() # instance of our CNN (random weights initially)
class_weights = torch.tensor([1.945, 0.673])
loss_function = nn.CrossEntropyLoss(weight = class_weights) # measures how wrong predictions are vs true labels
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001) # adjsuts mode's weights to reduce that error

num_epochs = 10 # how many full passes through the training data

for epoch in range(num_epochs):
    for images, labels in train_loader:  # one batch at a time (32 images + their true labels)
        optimizer.zero_grad() # clear old gradients from the previous batch
        outputs = model(images) # forward pass: get model's predictions (logits) for this batch
        loss = loss_function(outputs, labels) # compare predictions vs true labels -> single error number
        loss.backward() # backward pass: calculate how each weight should change
        optimizer.step() # actually update the weights using those calculations
        
    # Note: this only shows the loss form the LAST batch of the epoch, not the true average -
    #so values can look noisy/bouncy betwwen epoches. Fine for a first check , not a  reliable trend  
    print(f'Epoch {epoch+1}/{num_epochs}, Loss : {loss.item():.4f}')
    
# save the trained weights to a file so we don't have to retrain form scratch every time
#(this is the CENTRALIZED baseline model - will be compared against the federatede version later) 
torch.save(model.state_dict(), 'model_centralized.pth')
print('Model saved')

model.eval()

correct = 0
total = 0
normal_predicted = 0
pneumonia_predicted = 0 
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        normal_predicted += (predicted == 0).sum().item()
        pneumonia_predicted += (predicted == 1).sum().item()
accuracy = 100 * correct/total
print(f'Test Accuracy: {accuracy:.2f}')
print(f'Predicted Normal: {normal_predicted}')
print(f'Predicted PNEUMONIA: {pneumonia_predicted}')