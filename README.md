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

```cpp
class ParallelCNN_Transformer
{
public:
  // code for initializing and running the model
};
```

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