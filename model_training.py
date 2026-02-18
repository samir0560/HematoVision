import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
import os
from data_processing import BloodCellDataProcessor

class BloodCellClassifier:
    def __init__(self, num_classes=4, input_shape=(224, 224, 3)):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = None
        self.history = None
        self.classes = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']
        
    def build_model(self, learning_rate=0.001):
        """Build the MobileNetV2 transfer learning model"""
        print("🏗️  Building MobileNetV2 model...")
        
        # Load pre-trained MobileNetV2
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Add custom classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        
        # Create the model
        self.model = Model(inputs=base_model.input, outputs=predictions)
        
        # Compile the model
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("✅ Model built successfully!")
        return self.model
    
    def setup_callbacks(self):
        """Setup training callbacks"""
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                'models/best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        return callbacks
    
    def train_model(self, train_generator, validation_generator, epochs=25):
        """Train the model"""
        if self.model is None:
            self.build_model()
            
        print(f"🏋️  Training model for {epochs} epochs...")
        print("=" * 50)
        
        callbacks = self.setup_callbacks()
        
        self.history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        self.model.save('models/blood_model.h5')
        print("✅ Model saved as 'models/blood_model.h5'")
        
        return self.history
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available!")
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('static/training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def unfreeze_and_fine_tune(self, train_generator, validation_generator, 
                              fine_tune_epochs=10, fine_tune_at=100):
        """Unfreeze top layers and fine-tune the model"""
        print("🎯 Fine-tuning the model...")
        
        # Unfreeze the top layers of the base model
        base_model = self.model.layers[0]  # MobileNetV2 base
        base_model.trainable = True
        
        # Freeze all layers except the last ones
        for layer in base_model.layers[:-fine_tune_at]:
            layer.trainable = False
            
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=Adam(learning_rate=0.0001/10),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"Fine-tuning {len(base_model.trainable_variables)} layers")
        
        # Fine-tune training
        fine_tune_history = self.model.fit(
            train_generator,
            epochs=fine_tune_epochs,
            validation_data=validation_generator,
            verbose=1
        )
        
        # Save fine-tuned model
        self.model.save('models/blood_model_finetuned.h5')
        print("✅ Fine-tuned model saved as 'models/blood_model_finetuned.h5'")
        
        return fine_tune_history

def main():
    """Main training function"""
    print("🩸 HematoVision - Model Training")
    print("=" * 50)
    
    # Check if dataset exists
    if not os.path.exists('dataset/train') or not os.path.exists('dataset/test'):
        print("❌ Dataset not found!")
        print("Please organize your dataset as:")
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
        return
    
    # Initialize processor and classifier
    processor = BloodCellDataProcessor()
    classifier = BloodCellClassifier()
    
    # Create data generators
    print("📊 Creating data generators...")
    train_gen, val_gen = processor.create_data_generators(batch_size=32)
    
    # Build and train model
    classifier.build_model()
    history = classifier.train_model(train_gen, val_gen, epochs=25)
    
    # Plot results
    print("📊 Plotting training results...")
    classifier.plot_training_history()
    
    # Optional fine-tuning (uncomment if needed)
    # fine_tune_history = classifier.unfreeze_and_fine_tune(train_gen, val_gen, fine_tune_epochs=10)
    
    print("✅ Training completed successfully!")
    print("Model saved in 'models/' directory")

if __name__ == "__main__":
    main()