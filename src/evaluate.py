import torch
from model import PneumoniaCNN
from prep_data import test_loader

# Rebuild the mode structure, then load the already-trained weights into it
#(the .pth file only contains weights, not the articuture - that's why model.py is still needed)
model = PneumoniaCNN()
model.load_state_dict(torch.load('model_centralized.pth'))
model.eval()# switch to evaluation mode (no training happening here)

correct = 0
total = 0
normal_predicted = 0 # tracks how many images the model predicted as NORMAL
pneumonia_predicted = 0 # tracks how many images the model predicted as PNEUMONIA
# (used to check if mode is actually distinguishing both classes or just guessing one )

with torch.no_grad(): # no gradient tracking needed - we're only testing, not learning
    for images, labels in test_loader:
        outputs = model(images)  # forwars pass -> raw scores(logits) per image
        _, predicted = torch.max(outputs, 1) # pick the class with the higher score
        total += labels.size(0) # count images in this batch
        correct += (predicted == labels).sum().item() # count correct prediction
        normal_predicted += (predicted == 0).sum().item() # count NORMAL prediction
        pneumonia_predicted += (predicted == 1).sum().item() # count PNEUMONIA prediction
        
accuracy = 100 * correct / total
print(f'Test Accuracy : {accuracy:.2f}%')
print(f'Prediced NORMAL: {normal_predicted}')
print(f'Predicted PNEUMONIA: {pneumonia_predicted}')