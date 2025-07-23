import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
from .dcnn_model import GraspDCNN, GraspLoss
from .utils import load_config

class GraspDataset(Dataset):
    def __init__(self, dataset_path):
        with open(dataset_path, 'rb') as f:
            self.samples = pickle.load(f)
        print(f"Loaded {len(self.samples)} training samples")
        
        # Print dataset statistics
        qualities = [s['quality'] for s in self.samples]
        print(f"Quality range: {np.min(qualities):.3f} - {np.max(qualities):.3f}")
        print(f"Average quality: {np.mean(qualities):.3f}")
        
        # Count object types
        object_counts = {}
        for sample in self.samples:
            obj_name = sample['object_name']
            object_counts[obj_name] = object_counts.get(obj_name, 0) + 1
        
        print("Object distribution:")
        for obj, count in sorted(object_counts.items()):
            print(f"  {obj}: {count} samples ({count/len(self.samples)*100:.1f}%)")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Convert to tensors
        depth = torch.FloatTensor(sample['depth_image']).unsqueeze(0)  # Add channel dim
        grasp = torch.FloatTensor(sample['grasp_pose'])  # Already normalized
        
        # Data augmentation: random horizontal flip
        if np.random.random() > 0.5:
            depth = torch.flip(depth, [2])  # Flip horizontally
            grasp[0] = -grasp[0]  # Flip x-coordinate
            grasp[5] = -grasp[5]  # Flip yaw angle
        
        # Small random noise to depth image (regularization)
        if np.random.random() > 0.8:
            noise = torch.randn_like(depth) * 0.01
            depth = depth + noise
            depth = torch.clamp(depth, 0, 2.0)  # Keep reasonable depth range
        
        return depth, grasp

def plot_training_history(train_losses, val_losses, save_path='models/training_history.png'):
    """Plot and save training history"""
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(15, 5))
    
    # Total loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, [l[0] for l in train_losses], 'b-', label='Train Total')
    plt.plot(epochs, [l[0] for l in val_losses], 'r-', label='Val Total')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Position loss
    plt.subplot(1, 3, 2)
    plt.plot(epochs, [l[1] for l in train_losses], 'b-', label='Train Pos')
    plt.plot(epochs, [l[1] for l in val_losses], 'r-', label='Val Pos')
    plt.title('Position Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Rotation loss
    plt.subplot(1, 3, 3)
    plt.plot(epochs, [l[2] for l in train_losses], 'b-', label='Train Rot')
    plt.plot(epochs, [l[2] for l in val_losses], 'r-', label='Val Rot')
    plt.title('Rotation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training history saved to {save_path}")

def evaluate_model(model, val_loader, criterion, device):
    """Evaluate model on validation set"""
    model.eval()
    val_loss = 0
    val_pos_loss = 0
    val_rot_loss = 0
    num_samples = 0
    
    position_errors = []
    rotation_errors = []
    
    with torch.no_grad():
        for depth_images, grasp_poses in val_loader:
            depth_images = depth_images.to(device)
            grasp_poses = grasp_poses.to(device)
            
            predictions = model(depth_images)
            loss, pos_loss, rot_loss = criterion(predictions, grasp_poses)
            
            batch_size = depth_images.size(0)
            val_loss += loss.item() * batch_size
            val_pos_loss += pos_loss.item() * batch_size
            val_rot_loss += rot_loss.item() * batch_size
            num_samples += batch_size
            
            # Calculate actual errors for analysis
            pos_errors = torch.norm(predictions[:, :3] - grasp_poses[:, :3], dim=1)
            rot_errors = torch.norm(predictions[:, 3:] - grasp_poses[:, 3:], dim=1)
            
            position_errors.extend(pos_errors.cpu().numpy())
            rotation_errors.extend(rot_errors.cpu().numpy())
    
    # Average losses
    val_loss /= num_samples
    val_pos_loss /= num_samples
    val_rot_loss /= num_samples
    
    # Error statistics
    pos_error_stats = {
        'mean': np.mean(position_errors),
        'std': np.std(position_errors),
        'median': np.median(position_errors),
        'max': np.max(position_errors)
    }
    
    rot_error_stats = {
        'mean': np.mean(rotation_errors),
        'std': np.std(rotation_errors),
        'median': np.median(rotation_errors),
        'max': np.max(rotation_errors)
    }
    
    return (val_loss, val_pos_loss, val_rot_loss), pos_error_stats, rot_error_stats

def train_model(config_path="config/config.yaml", dataset_path="data/generated_training/training_dataset.pkl"):
    """Train the grasp prediction model with improvements"""
    config = load_config(config_path)
    train_config = config['training']
    
    # Setup device
    device = torch.device(train_config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Load and split dataset
    full_dataset = GraspDataset(dataset_path)
    
    # Split dataset (80% train, 20% validation)
    dataset_size = len(full_dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    
    torch.manual_seed(42)  # For reproducible splits
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=train_config['batch_size'], 
        shuffle=True,
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=train_config['batch_size'], 
        shuffle=False,
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    print(f"Dataset split: {len(train_dataset)} train, {len(val_dataset)} validation")
    
    # Initialize model
    model = GraspDCNN(config_path).to(device)
    criterion = GraspLoss(config_path)
    
    # Improved optimizer with stronger regularization
    optimizer = optim.Adam(
        model.parameters(), 
        lr=train_config['learning_rate'],
        weight_decay=1e-3,  # Increased weight decay to reduce overfitting
        eps=1e-8
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        patience=8,
        factor=0.5,
        min_lr=1e-6,
        #verbose=True
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Training tracking
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 15  # Early stopping patience
    
    print(f"\nStarting training for up to {train_config['epochs']} epochs...")
    print(f"Early stopping patience: {patience} epochs")
    print("-" * 60)
    
    # Training loop
    for epoch in range(train_config['epochs']):
        # Training phase
        model.train()
        train_loss = 0
        train_pos_loss = 0
        train_rot_loss = 0
        num_train_samples = 0
        
        for batch_idx, (depth_images, grasp_poses) in enumerate(train_loader):
            depth_images = depth_images.to(device, non_blocking=True)
            grasp_poses = grasp_poses.to(device, non_blocking=True)
            
            # Forward pass
            optimizer.zero_grad()
            predictions = model(depth_images)
            loss, pos_loss, rot_loss = criterion(predictions, grasp_poses)
            
            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Accumulate losses
            batch_size = depth_images.size(0)
            train_loss += loss.item() * batch_size
            train_pos_loss += pos_loss.item() * batch_size
            train_rot_loss += rot_loss.item() * batch_size
            num_train_samples += batch_size
            
            # Print progress
            if batch_idx % 20 == 0:
                print(f'Epoch {epoch+1:2d}/{train_config["epochs"]}, '
                      f'Batch {batch_idx:3d}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}, '
                      f'Pos: {pos_loss.item():.4f}, '
                      f'Rot: {rot_loss.item():.4f}')
        
        # Average training losses
        train_loss /= num_train_samples
        train_pos_loss /= num_train_samples
        train_rot_loss /= num_train_samples
        
        # Validation phase
        val_losses_epoch, pos_error_stats, rot_error_stats = evaluate_model(
            model, val_loader, criterion, device
        )
        val_loss, val_pos_loss, val_rot_loss = val_losses_epoch
        
        # Store losses for plotting
        train_losses.append((train_loss, train_pos_loss, train_rot_loss))
        val_losses.append((val_loss, val_pos_loss, val_rot_loss))
        
        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print epoch summary
        print(f'\nEpoch {epoch+1:2d} Summary:')
        print(f'  Train - Total: {train_loss:.4f}, Pos: {train_pos_loss:.4f}, Rot: {train_rot_loss:.4f}')
        print(f'  Val   - Total: {val_loss:.4f}, Pos: {val_pos_loss:.4f}, Rot: {val_rot_loss:.4f}')
        print(f'  Position Error - Mean: {pos_error_stats["mean"]:.4f}, Std: {pos_error_stats["std"]:.4f}')
        print(f'  Rotation Error - Mean: {rot_error_stats["mean"]:.4f}, Std: {rot_error_stats["std"]:.4f}')
        print(f'  Learning Rate: {current_lr:.6f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config
            }, 'models/best_model.pth')
            print(f'  ✓ New best model saved! (Val Loss: {val_loss:.4f})')
        else:
            patience_counter += 1
            print(f'  Patience: {patience_counter}/{patience}')
        
        # Early stopping check
        if patience_counter >= patience:
            print(f'\nEarly stopping triggered after {epoch + 1} epochs')
            print(f'Best validation loss: {best_val_loss:.4f}')
            break
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f'models/checkpoint_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
            }, checkpoint_path)
            print(f'  Checkpoint saved: {checkpoint_path}')
        
        print("-" * 60)
    
    # Load best model for final save
    print("\nLoading best model for final save...")
    best_checkpoint = torch.load('models/best_model.pth', map_location=device)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    # Save final model (just state dict for inference)
    torch.save(model.state_dict(), 'models/grasp_model_final.pth')
    
    # Plot and save training history
    plot_training_history(train_losses, val_losses)
    
    # Save training log
    training_log = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'final_epoch': len(train_losses),
        'config': config
    }
    
    with open('models/training_log.pkl', 'wb') as f:
        pickle.dump(training_log, f)
    
    print(f"\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final training loss: {train_losses[-1][0]:.4f}")
    print(f"Model saved as: models/grasp_model_final.pth")
    print(f"Best model saved as: models/best_model.pth")
    print(f"Training history plot: models/training_history.png")

def resume_training(checkpoint_path, config_path="config/config.yaml", 
                   dataset_path="data/generated_training/training_dataset.pkl"):
    """Resume training from a checkpoint"""
    print(f"Resuming training from {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)
    start_epoch = checkpoint['epoch']
    
    # Initialize everything
    config = load_config(config_path)
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    
    # ... (similar setup as train_model, but load checkpoint states)
    
    print(f"Resuming from epoch {start_epoch}")
    # Continue training loop from start_epoch...

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "resume":
        if len(sys.argv) > 2:
            resume_training(sys.argv[2])
        else:
            print("Usage: python -m src.training resume <checkpoint_path>")
    else:
        # Make sure models directory exists
        os.makedirs('models', exist_ok=True)
        
        # Run training
        train_model()
