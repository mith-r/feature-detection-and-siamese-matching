"""
Model architecture for Siamese Neural Network
"""
import torch
import torch.nn as nn
import torchvision

class Flatten(nn.Module):
    """Flatten layer to convert 4D tensor to 2D tensor."""
    def forward(self, input):
        return input.view(input.size(0), -1)

class SiameseNetwork(nn.Module):
    """
    Siamese Neural Network implementation
    
    Args:
        contra_loss (bool): Whether to use contrastive loss (True) or BCE loss (False)
    """
    def __init__(self, contra_loss=False):
        super(SiameseNetwork, self).__init__()
        self.contra_loss = contra_loss

        # Initialize the ResNet18 backbone
        # Hint: Use torchvision.models.resnet18 and modify it appropriately
        # 1. Initialize the ResNet18 model
        self.backbone = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)

        # 2. Modify the first convolutional layer if needed
        # ResNet18's default conv1 already accepts 3-channel RGB input
        # (Conv2d(3, 64, kernel_size=7, ...)), so no modification is needed.


        # 3. Store the number of features from the final layer
        self.num_features = self.backbone.fc.in_features

        # 4. Remove the final classification layer
        self.backbone.fc = nn.Identity()

        # Create additional layers for BCE loss
        # Hint: You need fully connected layers to process the concatenated features
        # and a sigmoid activation for the final output
        self.fc = nn.Sequential(
            nn.Linear(self.num_features * 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256,1),
            nn.Sigmoid(),
        )
        
        # Initialize the weights of your network
        # Hint: Create a method to initialize weights and apply it to your layers
        self.fc.apply(self.init_weights)

    
    def init_weights(self, m):
        # Implement weight initialization for linear layers
        # Hint: Use Xavier initialization for weights and small constant for biases
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.01)
    
    def forward_once(self, x):
        """
        Forward pass for one input image
        """
        # Implement the forward pass for a single image
        # The function should return the feature vector for the input image
        # Hint: Pass the input through the backbone network and flatten the output
        return self.backbone(x)
    
    def forward(self, input1, input2):
        """
        Forward pass for the Siamese network
        """
        # Implement the complete forward pass
        # 1. Get embeddings for both input images using forward_once
        # 2. Handle both cases: contrastive loss and BCE loss
        #    - For contrastive loss: return both embeddings
        #    - For BCE loss: concatenate embeddings, pass through FC layers, apply sigmoid

        out1 = self.forward_once(input1)
        out2 = self.forward_once(input2)

        if self.contra_loss:
            return out1, out2
        
        combined = torch.cat((out1, out2), dim = 1)
        return self.fc(combined)
        
