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


ImageClassifier/
│── Res18_model.py # ResNet-18 model definition
│── main_train.py # Training script
│── resnet18_best.pth # Saved model weights
│── README.md
│── res/
└── cifar-10/ # CIFAR-10 dataset (must be placed here)


---

## 📥 Dataset Setup

Download CIFAR-10 and place it in:


./res/cifar-10/


Expected structure:


res/
└── cifar-10/
├── train/
├── test/


> ⚠️ Make sure the dataset is correctly extracted and accessible.

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install torch torchvision
2. Train the model
python main_train.py
🧠 Model Details
Architecture: ResNet-18
Dataset: CIFAR-10 (10 classes)
Input Size: 32x32 RGB images
Loss Function: CrossEntropyLoss
Optimizer: (as defined in main_train.py)
💾 Pretrained Weights

The trained model is saved as:

resnet18_best.pth

You can load it using:

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
Ensure dataset path is correct (./res/cifar-10/)
GPU is recommended for faster training
Modify hyperparameters in main_train.py as needed
🛠️ Future Improvements
Add validation metrics visualization
Support for other architectures
Data augmentation
Training logs and checkpoints
📜 License

This project is for educational purposes.


---

If you want, I can :contentReference[oaicite:0]{index=0}—that really boosts impact when recruit# ImageClassifier - ResNet18 on CIFAR-10

This project implements a **ResNet-18** model from scratch using PyTorch for image classification on the **CIFAR-10** dataset.

---

## 📌 Features
- ResNet-18 architecture implementation
- Training pipeline for CIFAR-10
- Model checkpoint saving (`.pth`)
- Simple and modular code structure

---

## 📂 Project Structure


ImageClassifier/
│── Res18_model.py # ResNet-18 model definition
│── main_train.py # Training script
│── resnet18_best.pth # Saved model weights
│── README.md
│── res/
└── cifar-10/ # CIFAR-10 dataset (must be placed here)


---

## 📥 Dataset Setup

Download CIFAR-10 and place it in:


./res/cifar-10/


Expected structure:


res/
└── cifar-10/
├── train/
├── test/


> ⚠️ Make sure the dataset is correctly extracted and accessible.

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install torch torchvision
2. Train the model
python main_train.py
🧠 Model Details
Architecture: ResNet-18
Dataset: CIFAR-10 (10 classes)
Input Size: 32x32 RGB images
Loss Function: CrossEntropyLoss
Optimizer: (as defined in main_train.py)
💾 Pretrained Weights

The trained model is saved as:

resnet18_best.pth

You can load it using:

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
Ensure dataset path is correct (./res/cifar-10/)
GPU is recommended for faster training
Modify hyperparameters in main_train.py as needed
🛠️ Future Improvements
Add validation metrics visualization
Support for other architectures
Data augmentation
Training logs and checkpoints
📜 License

This project is for educational purposes.


---

If you want, I can :contentReference[oaicite:0]{index=0}—that really boosts impact when recruit
