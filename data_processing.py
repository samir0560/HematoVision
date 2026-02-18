import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class BloodCellDataProcessor:
    def __init__(self, dataset_path="dataset"):
        self.dataset_path = dataset_path
        self.classes = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']
        self.class_counts = {}
        
    def explore_dataset(self):
        """Explore and analyze the dataset structure"""
        print("🔍 Dataset Exploration")
        print("=" * 50)
        
        if not os.path.exists(self.dataset_path):
            print(f"Dataset folder '{self.dataset_path}' not found!")
            return False
            
        total_images = 0
        for class_name in self.classes:
            class_path = os.path.join(self.dataset_path, 'train', class_name)
            if os.path.exists(class_path):
                images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                count = len(images)
                self.class_counts[class_name] = count
                total_images += count
                print(f"{class_name:12}: {count:4} images")
            else:
                print(f"{class_name:12}: Folder not found")
                self.class_counts[class_name] = 0
                
        print(f"\nTotal images: {total_images}")
        return total_images > 0
    
    def visualize_data_distribution(self):
        """Create visualization of class distribution"""
        if not self.class_counts:
            print("No data to visualize!")
            return
            
        plt.figure(figsize=(12, 5))
        
        # Bar plot
        plt.subplot(1, 2, 1)
        classes = list(self.class_counts.keys())
        counts = list(self.class_counts.values())
        bars = plt.bar(classes, counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        plt.title('Blood Cell Class Distribution', fontsize=14, fontweight='bold')
        plt.ylabel('Number of Images')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        # Pie chart
        plt.subplot(1, 2, 2)
        plt.pie(counts, labels=classes, autopct='%1.1f%%', startangle=90,
                colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        plt.title('Class Distribution (%)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('static/data_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def show_sample_images(self, samples_per_class=3):
        """Display sample images from each class"""
        fig, axes = plt.subplots(len(self.classes), samples_per_class, 
                                figsize=(15, 12))
        fig.suptitle('Sample Blood Cell Images', fontsize=16, fontweight='bold')
        
        for i, class_name in enumerate(self.classes):
            class_path = os.path.join(self.dataset_path, 'train', class_name)
            if os.path.exists(class_path):
                images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                selected_images = random.sample(images, min(samples_per_class, len(images)))
                
                for j, img_name in enumerate(selected_images):
                    img_path = os.path.join(class_path, img_name)
                    try:
                        img = Image.open(img_path)
                        axes[i, j].imshow(img)
                        axes[i, j].set_title(f'{class_name}', fontweight='bold')
                        axes[i, j].axis('off')
                    except Exception as e:
                        print(f"Error loading {img_path}: {e}")
                        axes[i, j].text(0.5, 0.5, 'Error', ha='center', va='center')
                        axes[i, j].axis('off')
            else:
                for j in range(samples_per_class):
                    axes[i, j].text(0.5, 0.5, f'{class_name}\n(No data)', 
                                  ha='center', va='center')
                    axes[i, j].axis('off')
        
        plt.tight_layout()
        plt.savefig('static/sample_images.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def setup_data_augmentation(self):
        """Setup data augmentation for training"""
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest'
        )
        
        validation_datagen = ImageDataGenerator(rescale=1./255)
        
        return train_datagen, validation_datagen
    
    def create_data_generators(self, batch_size=32):
        """Create train and validation data generators"""
        train_datagen, validation_datagen = self.setup_data_augmentation()
        
        train_generator = train_datagen.flow_from_directory(
            os.path.join(self.dataset_path, 'train'),
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True
        )
        
        validation_generator = validation_datagen.flow_from_directory(
            os.path.join(self.dataset_path, 'test'),
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        return train_generator, validation_generator

def main():
    """Main function to demonstrate data processing"""
    print("🩸 HematoVision - Data Processing Module")
    print("=" * 50)
    
    processor = BloodCellDataProcessor()
    
    # Explore dataset
    if processor.explore_dataset():
        print("\n📊 Creating visualizations...")
        processor.visualize_data_distribution()
        processor.show_sample_images()
        print("✅ Data processing completed!")
    else:
        print("❌ Dataset not found. Please download the blood cell dataset.")
        print("Dataset should be organized as:")
        print("dataset/")
        print("  ├── train/")
        print("  │   ├── EOSINOPHIL/")
        print("  │   ├── LYMPHOCYTE/")
        print("  │   ├── MONOCYTE/")
        print("  │   └── NEUTROPHIL/")
        print("  └── test/")
        print("      ├── EOSINOPHIL/")
        print("      ├── LYMPHOCYTE/")
        print("      ├── MONOCYTE/")
        print("      └── NEUTROPHIL/")

if __name__ == "__main__":
    main()