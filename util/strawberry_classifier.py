
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os
import random
from collections import Counter

import camera.camera_helper as camera_helper
import util.datacube_extractor as datacube_extractor
import util.roi_extractor as roi_extractor


class HSIClassifier:
    """Main class for HSI classification with separate training and inference capabilities."""
    
    def __init__(self, num_classes=3, device=None):
        self.num_classes = num_classes
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = HSI3DCNN(num_classes).to(self.device)
        self.normalization_stats = None
        
    def normalize_cubes(self, cubes):
        """Normalize cubes and store normalization stats for later use."""
        mean = np.mean(cubes, axis=(0, 2, 3), keepdims=True)
        std = np.std(cubes, axis=(0, 2, 3), keepdims=True)
        self.normalization_stats = (mean, std)
        return (cubes - mean) / (std + 1e-6)
    
    def normalize_single_cube(self, cube):
        """Normalize a single cube dynamically."""
        # Calculate mean and std dynamically for the given cube
        mean = np.mean(cube, axis=(0, 1, 2), keepdims=True)
        std = np.std(cube, axis=(0, 1, 2), keepdims=True)

        # Ensure cube is float32 for compatibility
        cube = cube.astype(np.float32)

        return (cube - mean) / (std + 1e-6)
    
    def train(self, all_cubes, all_labels, epochs=300, test_size=0.2, batch_size=128):
        self.model.train()
        """Train the model with the given data."""
        # Normalize and prepare data
        all_cubes = self.normalize_cubes(all_cubes)
        cubes_train, cubes_test, labels_train, labels_test = train_test_split(
            all_cubes, all_labels, test_size=test_size, stratify=all_labels, random_state=42
        )
        
        train_dataset = HSIPixelDataset(cubes_train, labels_train, augment=True)
        test_dataset = HSIPixelDataset(cubes_test, labels_test, augment=False)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # Class weights
        y_all = train_dataset.labels
        class_weights = compute_class_weight('balanced', classes=np.unique(y_all), y=y_all)
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
        
        # Training setup
        loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)
        
        # Training loop
        best_acc = 0.0
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            for xb, yb in train_loader:
                xb = xb.unsqueeze(1).to(self.device)
                yb = yb.to(self.device)

                optimizer.zero_grad()
                out = self.model(xb)
                loss = loss_fn(out, yb)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            scheduler.step()
            avg_loss = running_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

            # Evaluation
            acc = self._evaluate(test_loader)
            print(f"Validation Accuracy: {acc:.4f}")

            # Save best model
            if acc > best_acc:
                best_acc = acc
                torch.save(self.model.state_dict(), "best_hsi_model.pth")
                print("✅ Saved new best model.")

        print("Training complete.")
        return best_acc
    
    def _evaluate(self, data_loader):
        """Internal evaluation method."""
        self.model.eval()
        all_preds, all_labels_eval = [], []
        with torch.no_grad():
            for xb, yb in data_loader:
                xb = xb.unsqueeze(1).to(self.device)
                preds = self.model(xb).argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels_eval.extend(yb.numpy())
        
        from sklearn.metrics import accuracy_score
        return accuracy_score(all_labels_eval, all_preds)
    
    def load_model(self, model_path):
        try:
            # Initialize the model architecture
            self.model = HSI3DCNN(self.num_classes).to(self.device)
            
            # Load the state dictionary
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)

            print(f"Model loaded successfully from {model_path}")
            print(f"Model architecture: {self.model}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def predict_cube(self, cube):
        """Predict class for a single hyperspectral cube."""
        # Normalize and prepare the cube dynamically
        cube = self.normalize_single_cube(cube)
        cube_tensor = torch.tensor(cube[np.newaxis, np.newaxis, :, :, :], dtype=torch.float32).to(self.device)
        
        # Predict
        with torch.no_grad():
            pred = self.model(cube_tensor).argmax(dim=1).cpu().numpy()
        
        return pred[0]  # Return single prediction
    
    
    def predict_cube_with_certainty(self, cube):
        self.model.eval()
        cube = self.normalize_single_cube(cube)
        cube_tensor = torch.tensor(cube[np.newaxis, np.newaxis, :, :, :], dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits = self.model(cube_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred = np.argmax(probs)
            certainty = probs[pred]

        return pred, certainty

    
    def predict_batch(self, cubes):
        """Predict classes for a batch of cubes."""
        if self.normalization_stats is None:
            raise ValueError("Model not trained or loaded with normalization stats")
            
        # Normalize all cubes
        normalized_cubes = np.array([self.normalize_single_cube(c) for c in cubes])
        cube_tensors = torch.tensor(normalized_cubes[:, np.newaxis, :, :, :], dtype=torch.float32).to(self.device)
        
        # Predict
        with torch.no_grad():
            preds = self.model(cube_tensors).argmax(dim=1).cpu().numpy()
        
        return preds
    
    def get_class_distribution(self, preds):
        """Get count of predictions per class."""
        return Counter(preds)
    

class HSI3DCNN(nn.Module):
    """3D CNN model for hyperspectral image classification."""
    def __init__(self, num_classes):
        super(HSI3DCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(32 * 1 * 1 * 56, 128),  # Adjust based on input size
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class HSIPixelDataset(Dataset):
    """Dataset class for HSI cubes with augmentation support."""
    def __init__(self, cubes, labels, augment=False):
        self.cubes = cubes.astype(np.float32)
        self.labels = np.array(labels).astype(np.int64)
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        cube = self.cubes[idx]
        label = self.labels[idx]
        if self.augment:
            cube = self.augment_cube(cube)
        return torch.tensor(cube.copy(), dtype=torch.float32), torch.tensor(label)

    def augment_cube(self, cube):
        """Apply random augmentations to the cube."""
        if random.random() < 0.5:
            cube = np.flip(cube, axis=1)  # horizontal flip
        if random.random() < 0.5:
            cube = np.flip(cube, axis=2)  # vertical flip
        if random.random() < 0.3:
            noise = np.random.normal(0, 0.01, cube.shape)
            cube = cube + noise
        return cube

def load_cube(file_path):
    """Load a hyperspectral cube from a file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cube file {file_path} does not exist.")
    try:
        return np.load(file_path)
    except Exception as e:
        raise ValueError(f"Error loading cube from {file_path}: {e}")

# Gets ROI from ui and sends prediction to ui
def call_prediction(ui_context, dpg, crop_path):
    dpg = ui_context["dpg"]

    save_path = datacube_extractor.extractor(ui_context, crop_path)
    cube = roi_extractor.load_roi(save_path)

    pred, certainty = make_prediction(cube=cube)
    dpg.configure_item("ai_result", default_value=f"The result of the classification is: {pred}")

 
## Takes in a cube (5 pixels x 224 bands x 5 bands) and uses a 3D CNN to classify whether
def make_prediction(cube):
    try:
        if cube.size == 0:
            raise ValueError(f"Loaded cube is empty.")
    except Exception as e:
        print(f"Error loading cube: {e}")
        return

    # Initialize and load the model
    classifier = HSIClassifier(num_classes=3)
    try:
        classifier.load_model("Resources/best_hsi_model.pth")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Run prediction with certainty
    try:
        pred, certainty = classifier.predict_cube_with_certainty(cube)
        print(f"Predicted class: {pred}")
        print(f"Prediction certainty: {certainty:.4f}")
    except Exception as e:
        print(f"Error during prediction: {e}")
        
    return pred, certainty




# if __name__ == "__main__":
#     test_crop_path = r"Cropped_20250509_151112\crop_000_roi_06.npy"  # or whatever input your extractor expects\
#     cube = load_cube(test_crop_path)
#     make_prediction(cube=cube)
