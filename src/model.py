import torch 
import torch.nn as nn

# CNN for binary classifcation : NORMAL vs PNEUMONIA ON grayscale chex x - rays.
# This is the CENTRALIZED BASELINE model - trained on all data together. 
# Later compared againsst the federated(FedAvg) version trained on split data.
class PneumoniaCNN(nn.Module):
    def __init__(self):
        super().__init__() # required: sets up nn.Moudels's internal layer- tracking system
        
        ## ----- Block 1 -------
        # in_channels = 1 because prep_data.py converts every image to grayscale(1 channel)
        # out_channels = 16 -> this layer learns 16 different simple patterns(edges, blobs, etc)
        self.conv1 = nn.Conv2d(in_channels = 1 , out_channels = 16, kernel_size = 3)
        
        ## ------ Block 2----------
        # in_channels = 16 because that's how many conv1 outputs
        # out_channels = 32 -> depper layer , more channels = more complex combination
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3)
        
        # Activation function , reused after every conv/linear layer
        # without this , stacking layers woruld be matematically pointless(no non-linearity).
        self.relu = nn.ReLU()
        
        # shrinks spatial size by half each time(keeps strongest signal in each 2X2 patch)
        # reused after both conv blocks
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        # Truns the final 3d features maps [32, 54, 54] into one flat row of numbers, 
        # since the Linear (fully - connected) layers below need 1D input per image
        self.flatten = nn.Flatten()
        
        # 93312 = 32 X 54 X 54 -> the exact flattened size after both conv* pool blocks
        # compresses that huge vector fown to 128 'decesion -relevant 'numbers
        self.fc1 = nn.Linear(in_features = 93312, out_features = 128)
        
        # Final layer : 2 outputs = one score (logit) per class (NORMAL , PNEUMONIA)
        # whichever score is higher = the model's predection
        self.fc2 = nn.Linear(in_features = 128 , out_features = 2)
        
    def forward(self, x):
        # x starts as a batch of images: [batch_size, 1, 244, 244]
        
        x = self.conv1(x) # detect simple patterns -> [batch, 16, 222, 222]
        x = self.relu(x) # non-linearity , shape unchanged
        x = self.pool(x) # shrinks size -> [batch, 32, 54, 54]
        
        x = self.conv2(x) # detect more complex patter -> [batch, 32, 109, 109]
        x = self.relu(x)
        x = self.pool(x) # shrinks size -> [batch, 32, 54, 54]
        
        x = self.flatten(x) # [batch, 93312]
        
        x = self.fc1(x) # compress -> [batch, 128]
        x = self.relu(x)
        
        x = self.fc2(x) # final scores -> [batch, 2]
        
        return x
    
    
if __name__ == '__main__':
    model = PneumoniaCNN()
    fake_batch = torch.randn(4, 1, 224, 224)
    output = model(fake_batch)
    print(output.shape)