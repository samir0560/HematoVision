# 🩸 HematoVision: Advanced Blood Cell Classification Using Transfer Learning

HematoVision is an advanced blood cell classification system that utilizes transfer learning with MobileNetV2 to accurately identify and classify different types of blood cells. This system provides a reliable and scalable tool for pathologists and healthcare professionals, ensuring precise and efficient blood cell classification.

## 🎯 Project Overview

This project aims to develop an accurate and efficient model for classifying blood cells by employing transfer learning techniques. Utilizing a dataset of annotated blood cell images, the system can classify cells into four distinct categories: Eosinophil, Lymphocyte, Monocyte, and Neutrophil.

## 🧬 Supported Blood Cell Types

- **Eosinophil** - Granulocyte with bi-lobed nucleus
- **Lymphocyte** - Small white blood cell  
- **Monocyte** - Largest type of white blood cell
- **Neutrophil** - Most abundant white blood cell

## 🛠️ Technology Stack

- **TensorFlow/Keras** - Deep learning framework
- **MobileNetV2** - Pre-trained CNN architecture
- **Flask** - Web application framework
- **Transfer Learning** - Leveraging pre-trained models
- **Data Augmentation** - Improving model generalization

## 📁 Project Structure

```
Hematovision_Project/
├── templates/
│   ├── home.html          # Main upload page
│   ├── result.html        # Prediction results page
│   ├── about.html         # About page
│   └── error.html         # Error handling page
├── static/
│   ├── uploads/           # Uploaded images storage
│   └── [generated files]  # Charts and visualizations
├── dataset/
│   ├── train/             # Training data
│   │   ├── EOSINOPHIL/
│   │   ├── LYMPHOCYTE/
│   │   ├── MONOCYTE/
│   │   └── NEUTROPHIL/
│   └── test/              # Testing data
│       ├── EOSINOPHIL/
│       ├── LYMPHOCYTE/
│       ├── MONOCYTE/
│       └── NEUTROPHIL/
├── models/                # Saved model files
├── app.py                 # Main Flask application
├── data_processing.py     # Data exploration and preprocessing
├── model_training.py      # Model building and training
├── predict.py             # Prediction utilities
└── requirements.txt       # Python dependencies
```

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- pip package manager

### Installation

1. **Clone or download the project**
   ```bash
   cd Hematovision_Project
   ```

2. **Install required dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Dataset Setup** (Optional for training)
   - Download the blood cell dataset from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/blood-cells/data)
   - Organize the dataset in the following structure:
   ```
   dataset/
     ├── train/
     │   ├── EOSINOPHIL/
     │   ├── LYMPHOCYTE/
     │   ├── MONOCYTE/
     │   └── NEUTROPHIL/
     └── test/
         ├── EOSINOPHIL/
         ├── LYMPHOCYTE/
         ├── MONOCYTE/
         └── NEUTROPHIL/
   ```

### Running the Application

1. **Start the Flask application**
   ```bash
   python app.py
   ```

2. **Access the application**
   Open your web browser and navigate to: `http://127.0.0.1:5000`

## 🧪 Usage

### Data Processing and Visualization
```bash
python data_processing.py
```
This script will:
- Explore the dataset structure
- Create visualizations of class distribution
- Display sample images from each class
- Set up data augmentation parameters

### Model Training
```bash
python model_training.py
```
This script will:
- Build the MobileNetV2 transfer learning model
- Train the model on your dataset
- Save the trained model
- Generate training history plots

### Web Application Features

1. **Home Page** (`/`)
   - Upload blood cell images
   - View system information
   - Access different sections

2. **Prediction** (`/predict`)
   - Upload image for classification
   - View prediction results with confidence scores
   - See the analyzed image

3. **About** (`/about`)
   - Project information
   - Technology stack details
   - Performance metrics

## 📊 Model Performance

The model achieves high accuracy through:
- Transfer learning from ImageNet pre-trained weights
- Data augmentation for improved generalization
- Proper train/validation split (80/20)
- Early stopping and learning rate scheduling
- Batch normalization and dropout for regularization

## 🏥 Applications

### Scenario 1: Automated Diagnostic Systems
Integration into clinical diagnostic systems for real-time blood analysis and report generation.

### Scenario 2: Remote Medical Consultations
Telemedicine platform integration for remote blood cell analysis and diagnosis.

### Scenario 3: Educational Tools
Interactive learning platform for medical students and laboratory technicians.

## 📈 Development Roadmap

- [x] Basic Flask application
- [x] Data processing and visualization
- [x] Model training with MobileNetV2
- [x] Web interface with upload functionality
- [ ] Model fine-tuning capabilities
- [ ] Performance metrics dashboard
- [ ] API endpoints for external integration
- [ ] Mobile-responsive design enhancements

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for improvements.

## 📄 License

This project is for educational and research purposes.

## 🙏 Acknowledgments

- Dataset provided by Paul Timothy Mooney on Kaggle
- MobileNetV2 architecture by Google
- TensorFlow and Keras teams for the excellent deep learning framework

---
*Developed with ❤️ for advancing medical diagnostics through AI*