# Bypassing QueueIT including CAPTCHA Recognition and Adversarial Robustness - Master's Thesis

This repository contains the complete implementation and analysis for a master's thesis project focused on CAPTCHA recognition using deep learning and evaluating model robustness against adversarial attacks.

## Project Overview

This project implements and evaluates various Convolutional Neural Network (CNN) architectures for CAPTCHA text recognition, with a particular focus on adversarial robustness. The research includes comprehensive analysis of different attack methods and their effectiveness against trained models.

## Project Structure

### Core Components

#### `/classification/` - Main Model Training and Evaluation
- **`best_model_training.py`** - Primary training script for the best-performing model
- **`hyperparameter_tuning_final.py`** - Comprehensive hyperparameter optimization
- **`model_architecture_tuning.py`** - Architecture comparison and selection
- **`predictBaseCNN.py`** - Model prediction and inference script

#### `/data/` - Dataset Organization
- **`train2/`** - Training dataset (1,929 images)
- **`val2/`** - Validation dataset (242 images) 
- **`test2/`** - Test dataset (241 images)
- **`test2_organized/`** - Test data organized by CAPTCHA type (checkered, dark_blue, light_blue)
- **`train_2_adversarial_examples/`** - Generated adversarial examples (836 images total)
  - FGSM attacks (epsilon: 1, 2, 3)
  - BIM attacks (epsilon: 0.1, 0.2, 0.5)
  - Gaussian noise (sigma: 0.1, 0.2, 0.25)
  - IFGS attacks (confidence: 0.7, 0.8, 0.9)
  - Smooth FGSM attacks (various sigma/epsilon combinations)

#### `/classification/Adversarial/` - Adversarial Attack Implementation
- **`FGSM.py`** - Fast Gradient Sign Method implementation
- **`BIM_AECAPTCHA.py`** - Basic Iterative Method for CAPTCHA attacks
- **`SmoothFGSM.py`** - Smooth variant of FGSM
- **`IFGS_IAN.py`** - Iterative Fast Gradient Sign with confidence targeting
- **`GaussianNoise.py`** - Gaussian noise-based attacks
- **`generate_adversarial_examples.py`** - Batch adversarial example generation
- **`test_transferability.py`** - Cross-model attack transferability testing

#### `/eda/` - Exploratory Data Analysis
- **`captcha_analysis/analyze_captcha_types.py`** - Comprehensive CAPTCHA dataset analysis
- **`captcha_analysis/results/`** - Analysis outputs including:
  - Character distribution plots
  - CAPTCHA type classification
  - Length distribution analysis
  - Background color analysis

#### `/tools/` - Utility Scripts
- **`moveImages.py`** - Dataset splitting and organization
- **`o_zero_correction.py`** - Automated O/0 character correction using Roboflow API
- **`predict_compare_api.py`** - Model vs API prediction comparison

#### `/queueit_automation/` - Queue Management System
- **`main.py`** - Multi-threaded queue automation for testing
- **`queue_it_purchase.py`** - Queue-it integration for web testing
- **`proxies.txt`** - Proxy configuration

#### `/scraping/` - Data Collection
- **`scrapeCaptchas.py`** - CAPTCHA image collection script

### Model Checkpoints and Results

#### Results and Analysis
- **`/classification/architectures_models_32_50epochs/`** - Performance comparison of the six different Architectures including architecture plots
- **`/classification/models_hyperparameter_tuning_results/`** - Hyperparameter tuning results of the three out of the six best Architectures
- **`/classification/Adversarial/transferability_results/plots`** - Cross-model attack transferability analysis

## Dependencies

Key libraries used:
- **TensorFlow/Keras** - Deep learning framework
- **OpenCV** - Image processing
- **NumPy** - Numerical computations
- **Matplotlib** - Visualization
- **Roboflow API** - External model comparison
- **Requests** - HTTP requests for scraping and automation

---
