# Queue-it Automation

A Python implementation for automating Queue-it challenges, including proof-of-work and image captcha solutions.

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

## Testing Captcha Solving

You can test the captcha solving functionality separately using the provided test script:

```bash
# Test with a local image file
python test_captcha_model.py --model best_double_conv_layers_model.keras --image path/to/captcha.png

# Test with base64 encoded image string
python test_captcha_model.py --model best_double_conv_layers_model.keras --base64 "base64_string_here"
```

## Configuration

The application uses environment variables for configuration:
- Configure proxy settings, API keys, and other parameters as needed

## Troubleshooting

If you encounter issues:
1. Check the logs for detailed error information
2. Verify that all dependencies are installed correctly
3. Ensure the captcha model is properly trained and compatible with the expected input format
4. Check network connectivity for any external API calls

## Features

- Automated navigation through Queue-it waiting rooms
- Support for various captcha types:
  - Image captchas (via AI model)
  - ReCAPTCHA (invisible and normal)
- Proof-of-Work challenge solving
- Queue position monitoring
- Proxy rotation handling
- Size limit configuration (to avoid wasting time in long queues)

## Requirements

- Python 3.6+
- Required packages: requests, uuid

## Directory Structure

```
queueit_automation/
├── queue_it_purchase.py - Main Queue-it automation class
├── main.py - Example usage
└── README.md - This file
```

## How to Use

1. Import the QueueItPurchase class from queue_it_purchase.py
2. Create a Purchase object that implements the required interface
3. Create a QueueItPurchase instance with your Purchase object
4. Handle the Queue-it page response
5. Process any captchas or challenges
6. Get a queue ID and poll the queue until you get through

See `main.py` for an example implementation.

## Customizing the AI Model

To use your own AI model for solving image captchas:

1. Look at the `solve_img_cap` method in `queue_it_purchase.py`
2. Replace the placeholder API call with your own model logic
3. Ensure your model returns the text from the captcha image

## Important Notes

- This code is intended for educational purposes
- You should comply with the terms of service of any website you interact with
- Queue-it is designed to ensure fair access to high-demand websites, please respect that purpose

## Example Integration

```python
from queue_it_purchase import QueueItPurchase

# Create a purchase object with the required interface
purchase = YourPurchaseClass(url)

# Create a QueueItPurchase instance
queue_it = QueueItPurchase(purchase)

# Check if the site uses Queue-it
success, _ = queue_it.get_queue()
if not success:
    if queue_it.QueueItEnabled:
        print("Queue-it is enabled but requires additional steps")
    else:
        print("Queue-it is not active on this site")
    return

# Set response data from the initial request
# Note: get_queue method already sets these values if it was successful
# queue_it.resptext = response_text
# queue_it.respurl = response_url
# queue_it.respstatcode = response_status_code

# Solve Queue-it
success, error = queue_it.solve_queue_it()

# Handle any challenges
if queue_it.imgcaptcha:
    # Solve image captcha
    queue_it.solve_queue_it_img()
    queue_it.solve_img_cap()
    queue_it.submit_img_cap()

if queue_it.pow:
    # Solve proof-of-work challenge
    queue_it.get_pow_challenge()
    solutions, _ = queue_it.solve_pow_challenge(queue_it.Input, queue_it.Complexity, queue_it.Runs)
    queue_it.Solution = solutions
    queue_it.submit_pow_solution()

# Get queue ID and poll the queue
queue_it.get_queue_id()
while True:
    success, _ = queue_it.poll_queue()
    if success:
        # Successfully passed the queue
        break
    time.sleep(5)
``` 