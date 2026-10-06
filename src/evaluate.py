import torch
from model import PneumoniaCNN
from prep_data import test_loader

model = PneumoniaCNN()
model.load_state_dict(torch.load('model_centralized.pth'))
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
        
accuracy = 100 * correct / total
print(f'Test Accuracy : {accuracy:.2f}%')
print(f'Prediced NORMAL: {normal_predicted}')
print(f'Predicted PNEUMONIA: {pneumonia_predicted}')