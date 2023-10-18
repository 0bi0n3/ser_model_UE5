# Model Selection and Initialization in Unreal Engine: A Study on Speech Emotion Recognition

## Overview

This study delves into the integration of Speech Emotion Recognition (SER) models within Unreal Engine for game development. Two main models were considered:

1. **Parallel 2D CNN - Transformer Encoder**: Chosen for its suitability in real-time gaming contexts.
2. **Wav2Vec2**: Used as a benchmark for performance evaluation.

The chosen model, Parallel 2D CNN - Transformer Encoder, was trained on the RAVDESS dataset.

## Model Architecture

The Parallel 2D CNN - Transformer Encoder works with a MEL spectrogram as input and processes it through two parallel paths:

1. **Transformer Path**
    - Max Pooling
    - 4-layer Transformer encoder
    - Mean operation
2. **Convolutional Path**
    - Conv2D blocks
    - Flattening

The outputs from these paths merge and go through a Linear layer, generating logits used for predictions and loss computation.

```python
class ParallelModel(nn.Module):
    def __init__(self,num_emotions):
        super().__init__()
        # conv block
        self.conv2Dblock = nn.Sequential(
            # 1. conv block
            nn.Conv2d(in_channels=1,
                       out_channels=16,
                       kernel_size=3,
                       stride=1,
                       padding=1
                      ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.3),
            # 2. conv block
            nn.Conv2d(in_channels=16,
                       out_channels=32,
                       kernel_size=3,
                       stride=1,
                       padding=1
                      ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),
            # 3. conv block
            nn.Conv2d(in_channels=32,
                       out_channels=64,
                       kernel_size=3,
                       stride=1,
                       padding=1
                      ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),
            # 4. conv block
            nn.Conv2d(in_channels=64,
                       out_channels=64,
                       kernel_size=3,
                       stride=1,
                       padding=1
                      ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3)
        )
        # Transformer block
        self.transf_maxpool = nn.MaxPool2d(kernel_size=[2,4], stride=[2,4])
        transf_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=512, dropout=0.4, activation='relu')
        self.transf_encoder = nn.TransformerEncoder(transf_layer, num_layers=4)
        # Linear softmax layer
        self.out_linear = nn.Linear(320,num_emotions)
        self.dropout_linear = nn.Dropout(p=0)
        self.out_softmax = nn.Softmax(dim=1)
    def forward(self,x):
        # conv embedding
        conv_embedding = self.conv2Dblock(x) #(b,channel,freq,time)
        #new_shape = (conv_embedding.size(0), -1)  # Keep the batch size, flatten the rest
        # Reshape the tensor
        #conv_embedding = conv_embedding.reshape(new_shape)
        conv_embedding = torch.flatten(conv_embedding, start_dim=1) # do not flatten batch dimension
        # transformer embedding
        x_reduced = self.transf_maxpool(x)
        x_reduced = torch.squeeze(x_reduced,1)
        x_reduced = x_reduced.permute(2,0,1) # requires shape = (time,batch,embedding)
        transf_out = self.transf_encoder(x_reduced)
        transf_embedding = torch.mean(transf_out, dim=0)
        # concatenate
        complete_embedding = torch.cat([conv_embedding, transf_embedding], dim=1)
        # final Linear
        output_logits = self.out_linear(complete_embedding)
        output_logits = self.dropout_linear(output_logits)
        output_softmax = self.out_softmax(output_logits)
        return output_logits, output_softmax
```
(code referenced from JOVANOVIC, K., 10 Nov, 2022-last update, Speech-Emotion-Classification-with-PyTorch .
Available: https://github.com/Data-Science-kosta/Speech-Emotion-Classification-with-PyTorch [05
Oct, 2023]).

## Audio Preprocessing in Unreal Engine

A Python script within Unreal Engine handles audio preprocessing 'offline', performing tasks like:

- Scanning for `.wav` files
- Using the librosa library for audio preprocessing
- Extracting Mel spectrograms and saving as `.npy` files

## Real-Time Processing and Inference

The current implementation is focused on offline preprocessing. Future work could include real-time audio capture and processing within Unreal Engine.

## Unreal Engine Setup and Plugin Configuration

A new project was initiated using the Third Person C++ template. The Neural Network Environment (NNE) plugin was enabled for interfacing with the neural network model.

## Model Performance Metrics

- **Training Accuracy**: ~60.48%
- **Validation Accuracy**: ~65.03%

## Limitations and Future Work

The observed accuracies indicate room for improvements. Possible future work includes:

- Incorporating a tailored dataset
- Extended training
- Refined hyperparameters

## Conclusion

The study lays the groundwork for implementing SER models in interactive gaming via Unreal Engine. It contributes significantly to the intersection of machine learning and game development, opening avenues for emotionally responsive and immersive gaming experiences.

---

For more detailed insights, please refer to the full paper:  
**AED (TC70025C) - Scientific Paper (A2) - Oberon Day-West #21501990**