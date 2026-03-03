import torch
import torch.nn as nn
import torchvision.models as models

class CNNLSTM(nn.Module):
    def __init__(self, num_classes=2):
        super(CNNLSTM, self).__init__()
        
        # 1. Feature Extractor (ResNet50)
        resnet = models.resnet50(pretrained=True)
        # Remove the last classification layer (fc)
        modules = list(resnet.children())[:-1] 
        self.cnn = nn.Sequential(*modules)
        
        # Freeze CNN weights to speed up training (Transfer Learning)
        for param in self.cnn.parameters():
            param.requires_grad = False
            
        # 2. Sequence Processor (LSTM)
        # ResNet50 outputs 2048 features
        self.lstm = nn.LSTM(input_size=2048, hidden_size=256, num_layers=2, batch_first=True)
        
        # 3. Classifier
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        # Input shape: (batch_size, seq_len, C, H, W)
        batch_size, seq_len, c, h, w = x.size()
        
        # Merge batch and sequence dimensions for CNN processing
        c_in = x.view(batch_size * seq_len, c, h, w)
        
        # Extract features
        c_out = self.cnn(c_in) # Output: (batch*seq, 2048, 1, 1)
        
        # Reshape back for LSTM: (batch, seq, 2048)
        lstm_in = c_out.view(batch_size, seq_len, -1)
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(lstm_in)
        
        # Take the output of the last frame in the sequence
        last_out = lstm_out[:, -1, :]
        
        # Final classification
        out = self.fc(last_out)
        return out