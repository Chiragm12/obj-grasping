import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
import os
from .dcnn_model import GraspDCNN, GraspLoss
from .utils import load_config

class GraspDataset(Dataset):
    def __init__(self, dataset_path):
        with open(dataset_path, 'rb') as f:
            self.samples = pickle.load(f)
        print(f"Loaded {len(self.samples)} training samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Convert to tensors
        depth = torch.FloatTensor(sample['depth_image']).unsqueeze(0)  # Add channel dim
        grasp = torch.FloatTensor(sample['grasp_pose'])
        
        return depth, grasp

def train_model(config_path="config/config.yaml", dataset_path="data/generated_training/training_dataset.pkl"):
    config = load_config(config_path)
    train_config = config['training']
    
    # Setup
    device = torch.device(train_config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")
    
    # Data with validation split
    dataset = GraspDataset(dataset_path)
    
    # Split dataset for validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=train_config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=train_config['batch_size'], shuffle=False)
    
    # Model with better optimizer
    model = GraspDCNN(config_path).to(device)
    criterion = GraspLoss(config_path)
    
    # Better optimizer settings
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(train_config['epochs']):
        # Training
        model.train()
        train_loss = 0
        train_pos_loss = 0
        train_rot_loss = 0
        
        for batch_idx, (depth_images, grasp_poses) in enumerate(train_loader):
            depth_images = depth_images.to(device)
            grasp_poses = grasp_poses.to(device)
            
            optimizer.zero_grad()
            
            predictions = model(depth_images)
            loss, pos_loss, rot_loss = criterion(predictions, grasp_poses)
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            train_pos_loss += pos_loss.item()
            train_rot_loss += rot_loss.item()
            
            if batch_idx % 20 == 0:
                print(f'Epoch {epoch+1}/{train_config["epochs"]}, '
                      f'Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}')
        
        # Validation
        model.eval()
        val_loss = 0
        val_pos_loss = 0
        val_rot_loss = 0
        
        with torch.no_grad():
            for depth_images, grasp_poses in val_loader:
                depth_images = depth_images.to(device)
                grasp_poses = grasp_poses.to(device)
                
                predictions = model(depth_images)
                loss, pos_loss, rot_loss = criterion(predictions, grasp_poses)
                
                val_loss += loss.item()
                val_pos_loss += pos_loss.item()
                val_rot_loss += rot_loss.item()
        
        # Average losses
        train_loss /= len(train_loader)
        train_pos_loss /= len(train_loader)
        train_rot_loss /= len(train_loader)
        
        val_loss /= len(val_loader)
        val_pos_loss /= len(val_loader)
        val_rot_loss /= len(val_loader)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}:')
        print(f'  Train - Total: {train_loss:.4f}, Pos: {train_pos_loss:.4f}, Rot: {train_rot_loss:.4f}')
        print(f'  Val   - Total: {val_loss:.4f}, Pos: {val_pos_loss:.4f}, Rot: {val_rot_loss:.4f}')
        print(f'  LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'models/best_model.pth')
            print(f'  New best model saved! (Val Loss: {val_loss:.4f})')
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'models/model_epoch_{epoch+1}.pth')
    
    # Save final model
    torch.save(model.state_dict(), 'models/grasp_model_final.pth')
    print("Training complete!")


if __name__ == "__main__":
    train_model()
