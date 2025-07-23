#!/usr/bin/env python3

import os
import sys
from src.data_generation import GraspDatasetGenerator  
from src.training import train_model
from src.grasp_execution import GraspExecutor

def main():
    print("PyBullet Grasp DCNN Training Pipeline")
    print("="*50)
    
    # Step 1: Generate training data
    print("\n1. GENERATING TRAINING DATA")
    print("-" * 30)
    
    if not os.path.exists("data/generated_training/training_dataset.pkl"):
        generator = GraspDatasetGenerator()
        try:
            samples = generator.generate_dataset(total_samples=3000)
        finally:
            generator.close()
    else:
        print("Training data already exists, skipping generation...")
    
    # Step 2: Train model
    print("\n2. TRAINING MODEL")
    print("-" * 30)
    
    os.makedirs('models', exist_ok=True)
    train_model()
    
    # Step 3: Test model
    print("\n3. TESTING MODEL")
    print("-" * 30)
    
    if os.path.exists('models/grasp_model_final.pth'):
        executor = GraspExecutor('models/grasp_model_final.pth')
        results, success_rate = executor.test_grasping(num_tests=10, gui=True)
        
        print(f"\nFinal Results:")
        print(f"Success Rate: {success_rate:.1f}%")
    else:
        print("No trained model found!")

if __name__ == "__main__":
    main()
