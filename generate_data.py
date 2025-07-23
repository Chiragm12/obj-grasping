from src.data_generation import GraspDatasetGenerator

def main():
    print('Generating improved dataset...')
    generator = GraspDatasetGenerator()
    
    try:
        samples = generator.generate_dataset(total_samples=2000)
        print(f'Successfully generated {len(samples)} samples')
    finally:
        generator.close()

if __name__ == "__main__":
    main()  # Fixed: Added parentheses
