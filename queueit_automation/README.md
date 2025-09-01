# Queue-it Automation

A Python implementation for automating Queue-it challenges, including proof-of-work and captcha solutions.

## Setup

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Ensure your Keras model for captcha solving is available in the project directory:
   - Default model path: `best_double_conv_layers_model.keras`

## Usage

Run the main application:

```bash
python main.py
```

## Configuration

The application uses environment variables for configuration:
- Configure proxy settings, API keys, and other parameters as needed

## Directory Structure

```
queueit_automation/
├── queue_it_purchase.py - Main Queue-it automation class
├── main.py - Example usage
└── README.md - This file
```
