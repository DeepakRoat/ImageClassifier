# ImageClassifier - ResNet18 on CIFAR-10

This project implements a **ResNet-18** model from scratch using PyTorch for image classification on the **CIFAR-10** dataset.

---

## 📌 Features
- ResNet-18 architecture implementation
- Training pipeline for CIFAR-10
- Model checkpoint saving (`.pth`)
- Simple and modular code structure

---

## 📂 Project Structure

```text
ImageClassifier/
│── Res18_model.py       # ResNet-18 model definition
│── main_train.py        # Training script
│── resnet18_best.pth    # Saved model weights
│── README.md
└── res/
    └── cifar-10-batches-py/ # CIFAR-10 dataset (must be placed here)
        ├── batches.meta
        ├── data_batch_1
        ├── data_batch_2
        ├── data_batch_3
        ├── data_batch_4
        ├── data_batch_5
        └── test_batch

📥 Dataset Setup

Download the Python version of CIFAR-10 and place the extracted folder in:
Plaintext

./res/cifar-10-batches-py/

    ⚠️ Make sure the dataset is correctly extracted so the data_batch files are directly accessible in that folder.

🚀 How to Run
1. Install dependencies
Bash

pip install torch torchvision

2. Train the model
Bash

python main_train.py

🧠 Model Details

    Architecture: ResNet-18

    Dataset: CIFAR-10 (10 classes)

    Input Size: 32x32 RGB images

    Loss Function: CrossEntropyLoss

    Optimizer: (as defined in main_train.py)

    Accuracy: 88% (Trained on 5,000 images, tested on 1,000 images)

💾 Pretrained Weights

The trained model is saved as:

resnet18_best.pth

You can load it using:
Python

model.load_state_dict(torch.load("resnet18_best.pth"))
model.eval()

📊 CIFAR-10 Classes

    Airplane

    Automobile

    Bird

    Cat

    Deer

    Dog

    Frog

    Horse

    Ship

    Truck

⚡ Notes

    Ensure the dataset path is correct (./res/cifar-10-batches-py/)

    GPU is recommended for faster training

    Modify hyperparameters in main_train.py as needed

🛠️ Future Improvements

    Add validation metrics visualization

    Support for other architectures

    Data augmentation

    Training logs and checkpoints

📜 License

This project is for educational purposes.
