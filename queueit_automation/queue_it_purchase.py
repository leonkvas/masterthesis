import re
import json
import time
import uuid
import base64
import hashlib
import urllib.parse
import requests
import numpy as np
import sys
import io
from typing import List, Dict, Union, Any, Optional, Tuple

# Import TensorFlow and Keras for image captcha solving
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from PIL import Image
except ImportError:
    print("TensorFlow, Keras or PIL not installed. Please install them using:")
    print("pip install tensorflow pillow")


class HashedSolution:
    """
    Represents a solution for a proof-of-work challenge
    """
    def __init__(self, postfix: int, hash: str):
        self.Postfix = postfix
        self.Hash = hash


class ImgCap:
    """
    Represents the data for an image captcha
    """
    def __init__(self, meta: str, key: str, session_id: str, challenge_details: str, image_base64: str):
        self.Meta = meta
        self.Key = key
        self.SessionId = session_id
        self.ChallengeDetails = challenge_details
        self.ImageBase64 = image_base64


class PowStruct:
    """
    Represents a proof-of-work challenge
    """
    def __init__(self, meta: str, session_id: str, parameters: Dict, challengedetails: str):
        self.Meta = meta
        self.SessionID = session_id
        self.Parameters = parameters
        self.Challengedetails = challengedetails


class SessionInfo:
    """
    Session information for Queue-it
    """
    def __init__(self, session_id: str, timestamp: str, checksum: str, source_ip: str, version: int):
        self.SessionID = session_id
        self.Timestamp = timestamp
        self.Checksum = checksum
        self.SourceIP = source_ip
        self.Version = version


class PowResponse:
    """
    Response for a proof-of-work challenge
    """
    def __init__(self, is_verified: bool, session_info: SessionInfo):
        self.IsVerified = is_verified
        self.SessionInfo = session_info


class QueueIDResponse:
    """
    Response containing a queue ID
    """
    def __init__(self, queue_id: str):
        self.QueueID = queue_id


class QueueTicket:
    """
    Ticket information for a queue
    """
    def __init__(self):
        self.QueueNumber = None
        self.UsersInLineAheadOfYou = None
        self.Progress = None
        self.SecondsToStart = 0
        self.QueuePaused = False


class QueueTexts:
    """
    Text information for a queue
    """
    def __init__(self):
        self.Header = ""


class QueueResp:
    """
    Response for a queue status check
    """
    def __init__(self):
        self.PageID = ""
        self.Ticket = QueueTicket()
        self.Texts = QueueTexts()


class QueueItPurchase:
    def __init__(self, purchase):
        """
        Initialize the QueueItPurchase instance
        
        Args:
            purchase: The Purchase object that contains session, url, and other data
        """
        self.Purchase = purchase
        self.resptext = ""
        self.respurl = ""
        self.respstatcode = ""
        self.Host = ""
        self.Customer_ID = ""
        self.EventId = ""
        self.QueueID = ""
        self.TargetUrl = ""
        self.captcha = False
        self.imgcaptcha = False
        self.pow = False
        self.Recapcaptcha = False
        self.sitekey = ""
        self.culture = ""
        self.customparams = ""
        self.challengeApiChecksumHash = ""
        self.beforeoridle = ""
        self.LayoutVersion = ""
        self.LayoutName = ""
        self.enqueuerequired = ""
        self.enqueuetoken = ""
        self.inviteonly = False
        self.queueitsolved = False
        self.sessioninfo = ""
        self.Solve = False
        self.AlternativePath = ""
        self.QueueItHeaderToken = ""
        self.QueueLiveTries = 0
        self.Tries = 0
        self.User_Agent = purchase.User_Agent if hasattr(purchase, 'User_Agent') else "Mozilla/5.0"
        self.SessionInfoIMG = ""
        self.SessionInfoRecapInvis = ""
        self.SessionInfoRecap = ""
        self.SessionInfoPOW = ""
        self.softblock = False
        self.QueueItEnabled = False
        self.QueueItCookie = ""
        
        # Image captcha related
        self.Meta = ""
        self.key = ""
        self.sessionid = ""
        self.ChallengeDetails = ""
        self.imgbase64 = ""
        self.response = ""
        self.SessionIdimgcap = ""
        self.Timestampimgcap = ""
        self.checksumimgcap = ""
        self.SourceIPimgcap = ""
        self.Versionimgcap = ""
        
        # POW related
        self.SessionId1 = ""
        self.Type = ""
        self.Input = ""
        self.Zerocount = 0
        self.Complexity = 0
        self.Runs = 0
        self.Solution = None
        self.SessionId2 = ""
        self.Timestamp2 = ""
        self.checksum2 = ""
        self.SourceIP2 = ""
        self.Version2 = 0
        
        # Queue polling
        self.Seid = None
        self.Sets = 0

    def re_find(self, text: str, pattern: str) -> str:
        """
        Find a pattern in text
        
        Args:
            text: The text to search in
            pattern: The regex pattern to search for
            
        Returns:
            The first match or empty string if not found
        """
        match = re.search(pattern, text)
        if match:
            return match.group(1)
        return ""
    
    def string_between(self, text: str, start: str, end: str) -> str:
        """
        Extract string between two substrings
        
        Args:
            text: The text to search in
            start: The starting substring
            end: The ending substring
            
        Returns:
            The string between start and end or empty string if not found
        """
        try:
            return text.split(start)[1].split(end)[0]
        except (IndexError, AttributeError):
            return ""
            
    def check_if_queue_has_started(self) -> None:
        """
        Check if the queue has started and set a global variable
        """
        # In the original code, this method sets a shared variable
        # We'll implement a placeholder for now
        pass
    
    def buy(self):
        """
        Main buying/initialization method. Loads the initial page and detects Queue-it
        """
        self.Purchase.info("Initiating Queue-it handler")
        
        # Reset basic values
        self.imgcaptcha = False 
        self.captcha = False
        self.Recapcaptcha = False
        self.pow = False
        self.softblock = False
        self.queueitsolved = False
        
        max_retries = 3  # Maximum number of retries for softblock
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                response = self.Purchase.Session.get(self.Purchase.Url)
                self.respurl = response.url
                self.respstatcode = str(response.status_code)
                self.resptext = response.text
                
                # Check for Queue-it in the page
                if "queue" in self.respurl.lower() and (self.Customer_ID in self.respurl.lower() and self.EventId in self.respurl.lower()):
                    # Queue detected
                    self.Purchase.info(f"Queue-it detected: {self.respurl}")
                    self.queueit = True
                    
                    # Check for softblock in the URL
                    if "softblock" in self.respurl.lower():
                        self.softblock = True
                        self.Purchase.error(f"Queue entry Softblocked! Retry {retry_count + 1}/{max_retries}")
                        
                        # Rotate proxy and create new session
                        self.Purchase.rotate_proxy()
                        self.Purchase.Session.delete_all_cookies()
                        
                        # Increment retry count
                        retry_count += 1
                        
                        # Sleep before retrying
                        time.sleep(3)
                        continue
                    
                    # Regular queue processing happens here (if not a softblock or after softblock handled)
                    if not self.queueitsolved and not self.softblock:
                        self.Purchase.info("Processing regular queue...")
                        success, error = self.solve_queue_it()
                        if not success:
                            self.Purchase.error(f"Failed to solve queue: {error}")
                            return False, error
                    
                    return True, None
                else:
                    # No Queue detected
                    self.Purchase.info("No Queue-it detected")
                    self.queueit = False
                    
                return True, None
            except Exception as e:
                self.Purchase.error(f"Exception in Queue-it buy method: {str(e)}")
                return False, e
        
        # If we've exhausted all retries
        self.Purchase.error(f"Failed to bypass softblock after {max_retries} attempts")
        return False, Exception("Max softblock retries exceeded")

    def solve_queue_it(self):
        """
        Main method to solve the Queue-it challenge
        
        Returns:
            Tuple containing success status (bool) and optional error
        """
        self.check_if_queue_has_started()
        
        if "decodeURIComponent" in self.resptext:
            self.TargetUrl = self.string_between(self.resptext, "decodeURIComponent('", "')")
            self.TargetUrl = urllib.parse.unquote(self.TargetUrl)
        
        if "/view?" in self.respurl:
            # NEW QUEUE LAYOUT
            self.Purchase.info("New Queue Layout")
            self.AlternativePath = "/queue"
        
        self.Host = self.re_find(self.respurl, r'https://(.*?)/')
        if not self.Host and "?" in self.respurl:
            self.Host = self.respurl.split('https://')[1]
            self.Host = self.Host.split('?')[0]
        
        if "?" in self.Host:
            self.Host = self.Host.split('?')[0]
        
        self.Customer_ID = self.re_find(self.resptext, r"customerId: '(.*?)'")
        if not self.Customer_ID:
            if "Error 403" in self.resptext:
                self.Purchase.error_with_ip_change("Error 403 -> Restarting task and changing IP..")
                self.Purchase.Session.delete_all_cookies()
                self.buy()
                time.sleep(9 * 60 * 60)  # Sleep for 9 hours (like in original code)
                return False, None
            else:
                self.Purchase.error_with_ip_change("QueueIt is not active on this Site. Retrying..")
                self.Purchase.Session.delete_all_cookies()
                self.buy()
                time.sleep(9 * 60 * 60)
                return False, None
        
        if '"name":"RecaptchaInvisible' in self.resptext:
            self.captcha = True
            self.sitekey = self.re_find(self.resptext, r"captchaInvisiblePublicKey: '(.*?)'")
            self.Purchase.info(f"ReCAPTCHA Invisible detected with sitekey: {self.sitekey}")
        else:
            self.captcha = False
        
        if '"name":"BotDetect' in self.resptext:
            self.imgcaptcha = True
            self.Purchase.info("Image captcha (BotDetect) detected")
        else:
            self.imgcaptcha = False
            
        if '"name":"ProofOfWork' in self.resptext:
            self.pow = True
            self.Purchase.info("Proof of Work challenge detected")
        else:
            self.pow = False
            
        if '"name":"Recaptcha' in self.resptext and not '"name":"RecaptchaInvisible' in self.resptext:
            self.Recapcaptcha = True
            # Set proper sitekey for Recaptcha
            self.sitekey = self.re_find(self.resptext, r"captchaPublicKey: '(.*?)'")
            if "shop.axs" in self.Purchase.Url:
                self.sitekey = "6Lc9sScUAAAAALTk003eM2ytnYGGKQaQa7usPKwo"
            elif not self.sitekey:
                self.sitekey = "6Lc9sScUAAAAALTk003eM2ytnYGGKQaQa7usPKwo"  # Default fallback sitekey
            self.Purchase.info(f"ReCAPTCHA detected with sitekey: {self.sitekey}")
        else:
            self.Recapcaptcha = False
            
        self.culture = self.re_find(self.resptext, r"culture: '(.*?)'")
        if not self.culture:
            self.culture = "en-US"
            
        self.customparams = self.re_find(self.resptext, r"customUrlParams: '(.*?)'")
        self.challengeApiChecksumHash = self.re_find(self.resptext, r"challengeApiChecksumHash: '(.*?)'")
        self.beforeoridle = self.re_find(self.resptext, r"isBeforeOrIdle: (.*?),")
        if not self.beforeoridle:
            self.beforeoridle = "true"
            
        if "layoutVersion" in self.resptext:
            self.LayoutVersion = self.string_between(self.resptext, "layoutVersion:", ",")
        else:
            self.LayoutVersion = self.re_find(self.resptext, r"layoutVersion:(.*?),")
            
        if "data-queueit-tag-eventid" in self.resptext:
            self.EventId = self.string_between(self.resptext, 'data-queueit-tag-eventid="', '"')
        else:
            self.EventId = self.re_find(self.resptext, r'data-queueit-tag-eventid="(.*?)"')
            
        if not self.EventId:
            # NEW QUEUE FORMAT
            self.EventId = self.re_find(self.resptext, r"eventId: '(.*?)'")
            
        if "data-queueit-tag-queueid" in self.resptext:
            self.QueueID = self.string_between(self.resptext, 'data-queueit-tag-queueid="', '"')
        else:
            self.QueueID = self.re_find(self.resptext, r'data-queueit-tag-queueid="(.*?)"')
            
        self.Purchase.ProductName = self.EventId
        self.Purchase.Image = self.re_find(self.resptext, r'src="(.*?)" id="imgCustomerLogo"')
        
        if not self.Purchase.Image.startswith("https:") and "//" in self.Purchase.Image and self.Purchase.Image:
            self.Purchase.Image = "https:" + self.Purchase.Image
        elif not self.Purchase.Image.startswith("https://") and self.Purchase.Image:
            self.Purchase.Image = "https://" + self.Purchase.Image
            
        if '"image","value":"' in self.resptext:
            self.Purchase.Image = self.re_find(self.resptext, r'"image","value":"(.*?)"')
            
        if "layoutName" in self.resptext:
            self.LayoutName = self.string_between(self.resptext, 'layoutName":"', '"')
        else:
            self.LayoutName = self.re_find(self.resptext, r'layoutName":"(.*?)"')
            
        if "isQueueitEnqueueTokenRequired" in self.resptext:
            self.enqueuerequired = self.string_between(self.resptext, "isQueueitEnqueueTokenRequired:", ",")
        else:
            self.enqueuerequired = self.re_find(self.resptext, r"isQueueitEnqueueTokenRequired: (.*?),")
            
        if "enqueuetoken" in self.respurl:
            self.enqueuetoken = self.re_find(self.respurl + "&", r"enqueuetoken=(.*?)&")
            if "&" in self.enqueuetoken:
                self.enqueuetoken = self.re_find(self.enqueuetoken, r"(.*?)&")
        
        #self.captcha = False
        #self.imgcaptcha = False
        #self.Recapcaptcha = False
        #self.pow = True

        if self.respstatcode == "200" and "c=" not in self.respurl:
            self.queueitsolved = True
            self.Purchase.success("Solved Queue-It")
            return True, None
        elif self.respstatcode == "200" and "c=" in self.respurl:
            self.queueitsolved = False
            if self.Recapcaptcha:
                print("Recap captcha detected, NOT IMPLEMENTED YET")
                return False, Exception("Recap captcha detected, NOT IMPLEMENTED YET")
            
            self.imgcaptcha = True
            if self.imgcaptcha:
                self.Purchase.info("Image captcha detected, starting solving process")
                
                # Try up to 3 times to solve the image captcha
                max_attempts = 3
                for attempt in range(1, max_attempts + 1):
                    self.Purchase.info(f"Image captcha solution attempt {attempt}/{max_attempts}")
                    
                    # First fetch the image captcha details
                    success, error = self.solve_queue_it_img()
                    if not success:
                        self.Purchase.error(f"Failed to fetch image captcha challenge: {error}")
                        if attempt == max_attempts:
                            return False, error
                        self.Purchase.info(f"Retrying in 2 seconds...")
                        time.sleep(2)
                        continue
                    
                    # Then solve the image captcha
                    self.Purchase.info("Solving image captcha...")
                    success, error = self.solve_img_cap()
                    if not success:
                        self.Purchase.error(f"Failed to solve image captcha: {error}")
                        if attempt == max_attempts:
                            return False, error
                        self.Purchase.info(f"Retrying in 2 seconds...")
                        time.sleep(2)
                        continue
                    
                    # Finally, submit the solution
                    self.Purchase.info("Submitting image captcha solution...")
                    success, error = self.submit_img_cap()
                    if not success:
                        self.Purchase.error(f"Failed to submit image captcha solution: {error}")
                        if attempt == max_attempts:
                            return False, error
                        self.Purchase.info(f"Retrying in 2 seconds...")
                        time.sleep(2)
                        continue
                    
                    # If we got here, we succeeded
                    self.Purchase.success(f"Successfully solved image captcha on attempt {attempt}")
                    break
                else:
                    # This only executes if the for loop completes without a break
                    self.Purchase.error("Failed to solve image captcha after maximum attempts")
                    return False, Exception("Failed after maximum retry attempts")

            sys.exit(0)  
            if self.captcha:
                print("Recap V3 Captcha detected, NOT IMPLEMENTED YET")
                return False, Exception("Recap V3 Captcha detected, NOT IMPLEMENTED YET")

            if self.pow:
                self.get_pow_challenge()
                solutions, err = self.solve_pow_challenge(self.Input, self.Complexity, self.Runs)
                if err:
                    self.Purchase.error("Error solving POW. Terminating Task.")
                    time.sleep(9 * 60 * 60)
                self.Solution = solutions
                self.submit_pow_solution()

            # After solving all challenges, get a queue ID
            if (self.pow or self.imgcaptcha or self.captcha or self.Recapcaptcha) and not self.softblock:
                self.Purchase.info("Getting queue ID after solving challenges")
                success, error = self.get_queue_id()
                if not success:
                    self.Purchase.error(f"Failed to get queue ID: {error}")
                    return False, error
                
                # Start polling the queue
                self.Purchase.info("Starting queue polling")
                return self.poll_queue()
                
            return True, None
        else:
            self.Purchase.error(f"Failed to solve Queue-It -- Unknown Error [{self.respstatcode}]")
            return False, None

    def solve_queue_it_img(self):
        """
        Handle image captcha challenge in Queue-it
        
        Returns:
            Tuple containing success status (bool) and optional error
        """
        self.check_if_queue_has_started()
        
        # Debug info
        self.Purchase.info(f"Fetching image captcha from: {self.Host}")
        self.Purchase.info(f"EventId: {self.EventId}")
        self.Purchase.info(f"Customer_ID: {self.Customer_ID}")
        self.Purchase.info(f"Alternative Path: {self.AlternativePath}")
        
        headers = {
            "sec-ch-ua": "\" Not A;Brand\";v=\"99\", \"Chromium\";v=\"100\", \"Google Chrome\";v=\"100\"",
            "x-queueit-challange-eventid": self.EventId,
            "x-queueit-challange-customerid": self.Customer_ID,
            "x-queueit-challange-hash": self.challengeApiChecksumHash,
            "x-queueit-challange-reason": "1",
            "sec-ch-ua-mobile": "?0",
            "User-Agent": self.User_Agent,
            "sec-ch-ua-platform": "\"macOS\"",
            "Accept": "*/*",
            "Origin": f"https://{self.Host}",
            "Sec-Fetch-Site": "same-origin",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Dest": "empty",
            "Referer": self.respurl,
            "Accept-Encoding": "gzip, deflate, br",
            "Accept-Language": "de-DE,de;q=0.9,en-US;q=0.8,en;q=0.7"
        }
        
        url = f"https://{self.Host}{self.AlternativePath}/challengeapi/queueitcaptcha/challenge/{self.culture}"
        self.Purchase.info(f"Image captcha request URL: {url}")
        
        try:
            response = self.Purchase.Session.post(url, headers=headers)
            
            self.Purchase.info(f"Image captcha response status: {response.status_code}")
            if response.status_code == 200:
                try:
                    self.Purchase.info(f"Image captcha response length: {len(response.text)}")
                    imgdata = json.loads(response.text)
                    
                    # Try to extract standard fields
                    self.Meta = imgdata.get('meta')
                    self.key = imgdata.get('key')
                    self.sessionid = imgdata.get('sessionId')
                    self.ChallengeDetails = imgdata.get('challengeDetails')
                    self.imgbase64 = imgdata.get('imageBase64')
                    
                    self.Purchase.info(f"Successfully fetched image captcha")
                    return True, None
                except json.JSONDecodeError as je:
                    self.Purchase.error(f"Failed to parse image captcha JSON response: {str(je)}")
                    self.Purchase.info(f"Response text: {response.text[:100]}...")
                    return False, Exception(f"JSON parse error: {str(je)}")
            elif response.status_code == 429:
                self.Purchase.error_with_ip_change("Error getting imgcap challenge -- Rate Limited -- Changing IP -- Retrying")
                return False, Exception("Rate limited 429")
            elif response.status_code == 404:
                self.Purchase.error_with_ip_change("Error getting imgcap challenge -- 404, Restarting Task.")
                self.Purchase.Session.delete_all_cookies()
                self.buy()
                time.sleep(9 * 60 * 60)  # Sleep for 9 hours
                return False, Exception("404 Not Found")
            else:
                response_text = response.text[:200] + "..." if len(response.text) > 200 else response.text
                self.Purchase.error_with_ip_change(f"Error getting imgcap challenge -- Unknown Error [{response.status_code}]")
                self.Purchase.error(f"Response: {response_text}")
                return False, Exception(f"HTTP error {response.status_code}")
        except Exception as e:
            self.Purchase.error(f"Exception in solve_queue_it_img: {str(e)}")
            return False, e
    
    # Define the custom metric function needed for the model
    def full_sequence_accuracy(self, y_true, y_pred):
        """
        Custom metric function needed for loading the Keras model
        
        Args:
            y_true: True labels
            y_pred: Predicted values
            
        Returns:
            Accuracy metric
        """
        y_pred_labels = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
        correct = tf.reduce_all(tf.equal(y_true, y_pred_labels), axis=1)
        return tf.reduce_mean(tf.cast(correct, tf.float32))

    def solve_img_cap(self):
        """
        Solve the image captcha using a local Keras model
        
        Returns:
            Tuple containing success status (bool) and optional error
        """
        self.check_if_queue_has_started()
        
        self.Purchase.interaction("Solving Image Captcha using local Keras model")
        
        try:
            # Check if required fields for image request are set
            if not self.Host:
                self.Purchase.error("Host is not set, cannot fetch image captcha")
                return False, Exception("Host is not set")
                
            if not self.EventId:
                self.Purchase.error("EventId is not set, cannot fetch image captcha")
                return False, Exception("EventId is not set")
                
            if not self.Customer_ID:
                self.Purchase.error("Customer_ID is not set, cannot fetch image captcha")
                return False, Exception("Customer_ID is not set")
        
            # Check if we need to fetch the image data first
            if not self.imgbase64:
                self.Purchase.info("No image data found, fetching image captcha first")
                success, error = self.solve_queue_it_img()
                if not success:
                    self.Purchase.error(f"Failed to fetch image captcha: {error}")
                    return False, error
                
                # Double-check that the image data was retrieved
                if not self.imgbase64:
                    self.Purchase.error("Failed to get image data after fetch attempt")
                    # Try to access the image captcha challenge directly as a fallback
                    self.Purchase.info("Attempting direct access to image captcha challenge")
                    direct_url = f"https://{self.Host}{self.AlternativePath}/challengeapi/queueitcaptcha/challenge/{self.culture}"
                    self.Purchase.interaction(f"Open this URL in browser to check what's happening: {direct_url}")
                    return False, Exception("Failed to get image data after fetch attempt")
            
            # Path to the Keras model
            model_path = "best_double_conv_layers_model.keras"
            
            # Try to load the model with custom metrics
            self.Purchase.info(f"Loading Keras model from {model_path}")
            model = load_model(model_path, custom_objects={'full_sequence_accuracy': self.full_sequence_accuracy})
            
            # Decode base64 image
            self.Purchase.info("Decoding base64 image")
            try:
                image_data = base64.b64decode(self.imgbase64)
            except Exception as e:
                self.Purchase.error(f"Failed to decode base64 image: {e}")
                self.Purchase.info(f"First 20 chars of imgbase64: {self.imgbase64[:20]}...")
                return False, e
            
            # Only save images in debug mode (remove automatic saving)
            if hasattr(self.Purchase, 'Debug') and self.Purchase.Debug:
                timestamp = int(time.time())
                captcha_filename = f"captcha_{timestamp}.jpg"
                try:
                    with open(captcha_filename, "wb") as f:
                        f.write(image_data)
                    self.Purchase.info(f"Saved captcha image to {captcha_filename} for verification")
                except Exception as e:
                    self.Purchase.error(f"Failed to save captcha image: {e}")
            else:
                captcha_filename = "captcha_image"  # Just a placeholder name for logs
            
            # Open the image using PIL
            img = Image.open(io.BytesIO(image_data))
            
            # Preprocess the image for the model
            # Note: Adjust these preprocessing steps based on your model's requirements
            self.Purchase.info("Preprocessing image")
            img = img.convert('L')  # Convert to grayscale
            img = img.resize((250, 50))  # Resize to expected dimensions (250x50)
            
            # Convert to numpy array and normalize
            img_array = np.array(img)
            img_array = img_array / 255.0  # Normalize to [0, 1]
            
            # Reshape for model input (add batch dimension and channel dimension if needed)
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            if len(img_array.shape) == 3:  # If grayscale, add channel dimension
                img_array = np.expand_dims(img_array, axis=-1)
            
            # Make prediction
            self.Purchase.info("Running model inference")
            prediction = model.predict(img_array)
            
            # Define the vocabulary exactly as in predictBaseCNN2.py
            vocab = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            idx_to_char = {i + 1: char for i, char in enumerate(vocab)}
            
            # Process prediction to get text
            # Convert predictions to indices (argmax over last dimension)
            pred_indices = np.argmax(prediction[0], axis=1)  # Take first batch item and get argmax per position
            
            # Convert indices to characters, skipping padding (0)
            result = ""
            for idx in pred_indices:
                if idx > 0:  # Skip padding tokens (0)
                    if idx in idx_to_char:
                        result += idx_to_char[idx]
            
            # Log debugging information
            self.Purchase.info(f"Raw prediction shape: {prediction.shape}")
            self.Purchase.info(f"Prediction indices: {pred_indices}")
            
            self.Purchase.info(f"Model predicted: {result}")
            if hasattr(self.Purchase, 'Debug') and self.Purchase.Debug:
                self.Purchase.interaction(f"CAPTCHA prediction for {captcha_filename}: '{result}' - If incorrect, please check the saved image")
            else:
                self.Purchase.interaction(f"CAPTCHA prediction: '{result}'")
            self.response = result
            
            if not self.response:
                self.Purchase.error("Model produced empty result")
                return False, Exception("Model produced empty result")
            
            return True, None
            
        except Exception as e:
            self.Purchase.error(f"Exception while solving captcha with Keras model: {str(e)}")
            return False, e
    
    def submit_img_cap(self):
        """
        Submit the solution for the image captcha
        
        Returns:
            Tuple containing success status (bool) and optional error
        """
        self.check_if_queue_has_started()
        
        self.Purchase.interaction("Submit Image Captcha")
        
        response_text = self.response
        self.Purchase.info(f"Submitting solution: '{response_text}'")
        
        if not self.sessionid:
            self.Purchase.error("Session ID is missing, cannot submit solution")
            return False, Exception("Session ID is missing")
            
        if not self.ChallengeDetails:
            self.Purchase.error("Challenge details are missing, cannot submit solution")
            self.Purchase.info("Using empty string for challenge details as fallback")
            self.ChallengeDetails = ""
        
        # Prepare the payload
        payload = {
            "challengeType": "botdetect",
            "sessionId": self.sessionid,
            "challengeDetails": self.ChallengeDetails,
            "solution": response_text,
            "stats": {
                "duration": 2473,
                "tries": 1,
                "userAgent": self.User_Agent,
                "screen": "1792 x 1120",
                "browser": "Chrome",
                "browserVersion": "108.0.0.0",
                "isMobile": False,
                "os": "Mac OS X",
                "osVersion": "10_15_7",
                "cookiesEnabled": True
            },
            "customerId": self.Customer_ID,
            "eventId": self.EventId,
            "version": 6
        }
        
        self.Purchase.info(f"Submission payload: customerId={self.Customer_ID}, eventId={self.EventId}, sessionId={self.sessionid}")
        
        headers = {
            "accept": "application/json, text/javascript, */*; q=0.01",
            "accept-encoding": "gzip, deflate, br",
            "accept-language": "en-US,en;q=0.9",
            "content-type": "application/json",
            "Origin": f"https://{self.Host}",
            "referer": self.Purchase.Url,
            "sec-ch-ua": "\" Not A;Brand\";v=\"99\", \"Chromium\";v=\"100\", \"Google Chrome\";v=\"100\"",
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": "\"macOS\"",
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "user-agent": self.User_Agent,
            "x-requested-with": "XMLHttpRequest",
            "x-queueit-challange-eventid": self.EventId,
            "x-queueit-challange-customerid": self.Customer_ID,
            "x-queueit-challange-hash": self.challengeApiChecksumHash
        }
        
        verify_url = f"https://{self.Host}{self.AlternativePath}/challengeapi/verify"
        self.Purchase.info(f"Submitting solution to: {verify_url}")
        
        try:
            response = self.Purchase.Session.post(
                verify_url,
                json=payload,
                headers=headers
            )
            
            self.Purchase.info(f"Submission response status: {response.status_code}")
            
            if response.status_code == 200:
                self.Purchase.info(f"Submission response: {response.text}")
                try:
                    response_data = json.loads(response.text)
                    
                    if response_data.get("IsVerified", False) or response_data.get("isVerified", False):
                        self.Purchase.success(f"Solved Image Captcha with solution: '{response_text}'")
                        
                        # Store the session info for later use in get_queue_id
                        session_info = response_data.get("sessionInfo", {})
                        self.SessionInfoIMG = {
                            "sessionId": session_info.get("sessionId", ""),
                            "timestamp": session_info.get("timestamp", ""),
                            "checksum": session_info.get("checksum", ""),
                            "sourceIp": session_info.get("sourceIp", ""),
                            "challengeType": "botdetect",
                            "version": session_info.get("version", 6),
                            "customerId": self.Customer_ID,
                            "waitingRoomId": self.EventId
                        }
                        
                        return True, None
                    else:
                        error_message = response_data.get("ErrorMessage", response_data.get("errorMessage", "Unknown error"))
                        self.Purchase.error(f"Image Captcha solution not accepted: {error_message}")
                        self.Purchase.rotate_proxy()
                        return False, Exception(f"Solution not accepted: {error_message}")
                except json.JSONDecodeError as e:
                    self.Purchase.error(f"Failed to parse solution response: {e}")
                    self.Purchase.error(f"Response text: {response.text[:200]}")
                    return False, e
                
            elif response.status_code in [429, 403]:
                self.Purchase.error(f"Error submitting Image Captcha Solution -- Status: {response.status_code}")
                if hasattr(response, 'text'):
                    self.Purchase.error(f"Response: {response.text[:200]}")
                return False, Exception(f"HTTP error {response.status_code}")
            else:
                self.Purchase.error(f"Error submitting Image Captcha Solution -- Unknown Error [{response.status_code}]")
                if hasattr(response, 'text'):
                    self.Purchase.error(f"Response: {response.text[:200]}")
                return False, Exception(f"HTTP error {response.status_code}")
                
        except Exception as e:
            self.Purchase.error(f"Exception in submit_img_cap: {str(e)}")
            return False, e

    def get_pow_challenge(self):
        """
        Get a proof-of-work challenge from the server
        
        Returns:
            Tuple containing success status (bool) and optional error
        """
        self.check_if_queue_has_started()
        
        # Check if we have the required parameters
        if not self.Host:
            self.Purchase.error("Host is not set, cannot get PoW challenge")
            return False, Exception("Host is not set")
        
        if not hasattr(self, 'QueueItCookie') or not self.QueueItCookie:
            self.Purchase.info("QueueItCookie is not set, using empty value")
            self.QueueItCookie = ""
        
        if not self.EventId:
            self.Purchase.error("EventId is not set, cannot get PoW challenge")
            return False, Exception("EventId is not set")
        
        if not self.Customer_ID:
            self.Purchase.error("Customer_ID is not set, cannot get PoW challenge")
            return False, Exception("Customer_ID is not set")
            
        self.Purchase.info(f"Getting PoW challenge from {self.Host}")
        self.Purchase.info(f"EventId: {self.EventId}")
        self.Purchase.info(f"Customer_ID: {self.Customer_ID}")
        
        headers = {
            "sec-ch-ua": "\" Not A;Brand\";v=\"99\", \"Chromium\";v=\"100\", \"Google Chrome\";v=\"100\"",
            "powTag-UserId": self.QueueItCookie,
            "powTag-EventId": self.EventId,
            "powTag-CustomerId": self.Customer_ID,
            "x-queueit-challange-eventid": self.EventId,
            "x-queueit-challange-customerid": self.Customer_ID,
            "x-queueit-challange-hash": self.challengeApiChecksumHash,
            "x-queueit-challange-reason": "1",
            "sec-ch-ua-mobile": "?0",
            "User-Agent": self.User_Agent,
            "x-queueit-challange-userid": self.QueueItCookie,
            "sec-ch-ua-platform": "\"macOS\"",
            "Accept": "*/*",
            "Origin": f"https://{self.Host}",
            "Sec-Fetch-Site": "same-origin",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Dest": "empty",
            "Referer": self.respurl,
            "Accept-Encoding": "gzip, deflate, br",
            "Accept-Language": "de-DE,de;q=0.9,en-US;q=0.8,en;q=0.7"
        }
        
        try:
            url = f"https://{self.Host}{self.AlternativePath}/challengeapi/pow/challenge/{self.QueueItCookie}"
            self.Purchase.info(f"POW challenge URL: {url}")
            
            response = self.Purchase.Session.post(url, headers=headers)
            
            self.Purchase.info(f"POW challenge response code: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    #self.Purchase.info(f"POW challenge response: {response.text}")
                    powdata = json.loads(response.text)
                    
                    #print(powdata)
                    self.Meta = powdata.get("meta", "")
                    self.SessionId1 = powdata.get("sessionId", "")
                    
                    parameters = powdata.get("parameters", {})
                    self.Type = parameters.get("type", "")
                    self.Input = parameters.get("input", "")
                    self.Zerocount = parameters.get("zeroCount", 0)
                    self.Complexity = parameters.get("complexity", 0)
                    self.Runs = parameters.get("runs", 0)
                    
                    self.ChallengeDetails = powdata.get("challengeDetails", "")
                    
                    self.Purchase.info(f"Extracted PoW challenge parameters:")
                    self.Purchase.info(f"- Type: {self.Type}")
                    self.Purchase.info(f"- Input: {self.Input}")
                    self.Purchase.info(f"- Complexity: {self.Complexity}")
                    self.Purchase.info(f"- Runs: {self.Runs}")
                    
                    return True, None
                except json.JSONDecodeError:
                    self.Purchase.error("Error parsing PoW challenge response JSON")
                    self.Purchase.error(f"Response text: {response.text}")
                    return False, Exception("Error parsing PoW challenge response JSON")
            elif response.status_code == 429:
                self.Purchase.error_with_ip_change("Error getting PoW challenge -- Rate Limited -- Changing IP -- Retrying")
                return False, None
            elif response.status_code == 404:
                self.Purchase.error_with_ip_change("Error getting PoW challenge -- 404, Restarting Task.")
                self.Purchase.Session.delete_all_cookies()
                self.buy()
                time.sleep(9 * 60 * 60)  # Sleep for 9 hours
                return False, None
            else:
                self.Purchase.error_with_ip_change(f"Error getting PoW challenge -- Unknown Error [{response.status_code}]")
                self.Purchase.error(f"Response text: {response.text if hasattr(response, 'text') else 'No text available'}")
                return False, None
        except Exception as e:
            self.Purchase.error(f"Exception in get_pow_challenge: {str(e)}")
            return False, e
    
    def solve_pow_challenge(self, input_str, complexity, runs):
        """
        Solve a proof-of-work challenge
        
        Args:
            input_str: The input string for the challenge
            complexity: The number of leading zeros required
            runs: The number of solutions required
            
        Returns:
            A list of HashedSolution objects or an error
        """
        self.check_if_queue_has_started()
        
        self.Purchase.info(f"Solving PoW challenge - input: {input_str}, complexity: {complexity}, runs: {runs}")
        
        solutions = []
        current_runs = 0
        postfix = 0
        
        try:
            while current_runs < runs:
                # Create input for hash
                input_data = input_str + str(postfix)
                
                # Calculate hash
                sha_obj = hashlib.sha256()
                sha_obj.update(input_data.encode())
                hash_hex = sha_obj.hexdigest()
                
                # Check if hash has required leading zeros
                if hash_hex.startswith("0" * complexity):
                    solution = HashedSolution(postfix, hash_hex)
                    solutions.append(solution)
                    current_runs += 1
                    self.Purchase.info(f"Found solution {current_runs}/{runs}: postfix={postfix}, hash={hash_hex}")
                
                postfix += 1
                
                # Safety check to prevent infinite loops
                if postfix > 10000000:  # 10 million attempts should be enough
                    break
            
            if current_runs == runs:
                # Convert solutions to a format that can be serialized to JSON
                serializable_solutions = []
                for solution in solutions:
                    serializable_solutions.append({
                        "Postfix": solution.Postfix,
                        "Hash": solution.Hash
                    })
                return serializable_solutions, None
            else:
                return [], Exception(f"Failed to solve PoW after {postfix} attempts")
        except Exception as e:
            self.Purchase.error(f"Error in solve_pow_challenge: {str(e)}")
            return [], e
    
    def submit_pow_solution(self):
        """
        Submit the solution for the proof-of-work challenge
        
        Returns:
            Tuple containing success status (bool) and optional error
        """
        self.check_if_queue_has_started()
        
        # Debug information about solution
        if not self.Solution:
            self.Purchase.error("No solution to submit")
            return False, Exception("No solution to submit")
        
        # Make sure solution is properly formatted
        if isinstance(self.Solution, list):
            # The Solution is already a list of dictionaries, so just convert to JSON string
            solution_str = json.dumps(self.Solution)
        else:
            # Assume it's already a JSON string
            solution_str = self.Solution
            
        self.Purchase.info(f"Solution type: {type(self.Solution)}")
        self.Purchase.info(f"Submitting solution: {solution_str}")
        
        # Create the solution data structure exactly matching the example
        solution_data = {
            "hash": self.Solution,  # Keep the original solution format
            "type": "HashChallenge"  # Use the exact type from example
        }
        
        # Convert to JSON string and then base64 encode
        solution_json = json.dumps(solution_data, separators=(',', ':'))
        encoded_solution = base64.b64encode(solution_json.encode()).decode()
        
        # Create the verification payload matching the example exactly
        payload = {
            "challengeType": "proofofwork",
            "sessionId": self.SessionId1,
            "challengeDetails": self.ChallengeDetails,
            "solution": encoded_solution,
            "stats": {
                "duration": 1623,  # Match example duration
                "tries": 1,
                "userAgent": self.User_Agent,
                "screen": "2560 x 1440",  # Match example screen size
                "browser": "Chrome",
                "browserVersion": "134.0.0.0",  # Match example version
                "isMobile": False,
                "os": "Mac OS X",
                "osVersion": "10_15_7",
                "cookiesEnabled": True
            },
            "customerId": self.Customer_ID,
            "eventId": self.EventId,
            "version": 6
        }
        
        self.Purchase.info(f"POW challenge payload:")
        self.Purchase.info(f"- customerId: {self.Customer_ID}")
        self.Purchase.info(f"- eventId: {self.EventId}")
        self.Purchase.info(f"- sessionId: {self.SessionId1}")
        self.Purchase.info(f"- encoded solution: {encoded_solution[:50]}...")
        
        headers = {
            "accept": "application/json, text/javascript, */*; q=0.01",
            "accept-encoding": "gzip, deflate, br",
            "accept-language": "en-US,en;q=0.9",
            "content-type": "application/json",
            "Origin": f"https://{self.Host}",
            "referer": self.Purchase.Url,
            "sec-ch-ua": "\" Not A;Brand\";v=\"99\", \"Chromium\";v=\"100\", \"Google Chrome\";v=\"100\"",
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": "\"macOS\"",
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "user-agent": self.User_Agent,
            "x-requested-with": "XMLHttpRequest",
            "x-queueit-challange-eventid": self.EventId,
            "x-queueit-challange-customerid": self.Customer_ID,
            "x-queueit-challange-hash": self.challengeApiChecksumHash
        }
        
        try:
            verify_url = f"https://{self.Host}{self.AlternativePath}/challengeapi/verify"
            self.Purchase.info(f"Sending POW solution to: {verify_url}")
            
            response = self.Purchase.Session.post(
                verify_url,
                json=payload,
                headers=headers
            )
            
            self.Purchase.info(f"POW submission response code: {response.status_code}")
            self.Purchase.info(f"POW submission response: {response.text}")
            
            if response.status_code == 200:
                try:
                    response_data = json.loads(response.text)
                    
                    if response_data.get("IsVerified", False) or response_data.get("isVerified", False):
                        # Store the session info for later use in get_queue_id
                        session_info = response_data.get("sessionInfo", {})
                        self.SessionInfoPOW = {
                            "sessionId": session_info.get("sessionId", ""),
                            "timestamp": session_info.get("timestamp", ""),
                            "checksum": session_info.get("checksum", ""),
                            "sourceIp": session_info.get("sourceIp", ""),
                            "challengeType": "proofofwork",
                            "version": session_info.get("version", 6),
                            "customerId": self.Customer_ID,
                            "waitingRoomId": self.EventId
                        }
                        
                        return True, None
                    else:
                        self.Purchase.error("PoW Solution not accepted -- Resolving")
                        self.get_pow_challenge()
                        self.Purchase.interaction("Solving PoW")
                        
                        solutions, err = self.solve_pow_challenge(self.Input, self.Complexity, self.Runs)
                        if err:
                            self.Purchase.error("Error solving POW. Terminating Task.")
                            time.sleep(9 * 60 * 60)
                        
                        self.Solution = solutions
                        self.submit_pow_solution()
                        
                        return True, None
                except json.JSONDecodeError:
                    self.Purchase.error("Error parsing POW response JSON")
                    return False, Exception("Error parsing POW response JSON")
            elif response.status_code == 429:
                self.Purchase.error_with_ip_change("Error submitting PoW Solution -- Rate Limited -- Changing IP -- Retrying")
                return False, None
            else:
                self.Purchase.error(f"Error submitting PoW Solution -- Unknown Error [{response.status_code}]")
                if response.text:
                    self.Purchase.error(f"Error details: {response.text}")
                return False, None
        except Exception as e:
            self.Purchase.error(f"Exception in submit_pow_solution: {str(e)}")
            return False, e

    def check_valid_session_info(self):
        """
        Verify that we have valid session information before making a queue ID request
        """
        if not self.SessionInfoPOW:
            self.Purchase.error("No POW session info available")
            return False
            
        # Check if the session info contains required fields
        try:
            # Minimal validation to check if it contains key fields
            required_fields = ["sessionId", "timestamp", "checksum"]
            for field in required_fields:
                if field not in self.SessionInfoPOW:
                    self.Purchase.error(f"Missing required field '{field}' in session info")
                    return False
            return True
        except Exception as e:
            self.Purchase.error(f"Error validating session info: {str(e)}")
            return False

    def get_queue_id(self):
        try:
            # Prepare the request URL
            queue_id_url = f"https://{self.Host}/spa-api/queue/{self.Customer_ID}/{self.EventId}/enqueue"
            
            # Add culture parameter
            if self.culture:
                queue_id_url += f"?cid={self.culture}"
            
            self.Purchase.info(f"Queue ID request URL: {queue_id_url}")
            
            # Prepare challenge sessions array
            challenge_sessions = []
            
            # Add Image Captcha session if available
            if hasattr(self, 'SessionInfoIMG') and self.SessionInfoIMG:
                challenge_sessions.append(self.SessionInfoIMG)
            
            # Add POW session if available
            if hasattr(self, 'SessionInfoPOW') and self.SessionInfoPOW:
                challenge_sessions.append(self.SessionInfoPOW)
            
            # Prepare the payload with the new format
            payload = {
                "challengeSessions": challenge_sessions,
                "layoutName": self.LayoutName,
                "customUrlParams": self.customparams,
                "targetUrl": self.TargetUrl,
                "Referrer": self.respurl
            }
            
            # Prepare headers
            headers = {
                "accept": "application/json, text/javascript, */*; q=0.01",
                "content-type": "application/json",
                "origin": f"https://{self.Host}",
                "referer": self.respurl,
                "sec-ch-ua": "\"Google Chrome\";v=\"135\", \"Not-A.Brand\";v=\"8\", \"Chromium\";v=\"135\"",
                "sec-ch-ua-mobile": "?0",
                "sec-ch-ua-platform": "\"macOS\"",
                "sec-fetch-dest": "empty",
                "sec-fetch-mode": "cors",
                "sec-fetch-site": "same-origin",
                "user-agent": self.User_Agent,
                "x-requested-with": "XMLHttpRequest",
                "x-queueit-qpage-referral": self.respurl,
                "priority": "u=1, i"
            }
        
            
            # Make the request
            response = self.Purchase.Session.post(
                queue_id_url,
                json=payload,
                headers=headers
            )
            
            self.Purchase.info(f"Queue ID response status: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    response_data = json.loads(response.text)
                    
                    # Extract queue ID from response
                    if "queueId" in response_data:
                        self.QueueID = response_data["queueId"]
                        self.Purchase.success(f"Successfully obtained Queue ID: {self.QueueID}")
                        
                        # Generate unique session values
                        self.Seid = str(uuid.uuid4())
                        self.Sets = str(int(time.time() * 1000))  # Current time in milliseconds
                        
                        return True, None
                    else:
                        self.Purchase.error("Queue ID not found in response")
                        self.Purchase.debug(f"Response: {response.text[:200]}...")
                        return False, Exception("Queue ID not found in response")
                except json.JSONDecodeError:
                    self.Purchase.error("Failed to parse queue ID response")
                    self.Purchase.debug(f"Response: {response.text[:200]}...")
                    return False, Exception("Failed to parse queue ID response")
            elif response.status_code == 429:
                self.Purchase.error("Rate limited while getting queue ID")
                return False, Exception("Rate limited")
            else:
                self.Purchase.error(f"Failed to get queue ID - Status code: {response.status_code}")
                self.Purchase.debug(f"Response: {response.text[:200]}...")
                return False, Exception(f"Failed to get queue ID - Status code: {response.status_code}")
                
        except Exception as e:
            self.Purchase.error(f"Exception in get_queue_id: {str(e)}")
            return False, Exception(f"Exception in get_queue_id: {str(e)}")
    
    def poll_queue(self):
        """
        Poll the queue status
        
        Returns:
            Tuple containing success status (bool) and optional error
        """
        # Payload for the queue status request
        payload = {
            "targetUrl": self.TargetUrl,
            "customUrlParams": self.customparams,
            "layoutVersion": self.LayoutVersion,
            "layoutName": self.LayoutName,
            "isClientRedayToRedirect": True,
            "isBeforeOrIdle": False
        }
        
        headers = {
            "accept": "application/json, text/javascript, */*; q=0.01",
            "accept-encoding": "gzip, deflate, br",
            "accept-language": "en-US,en;q=0.9",
            "content-type": "application/json",
            "origin": f"https://{self.Host}",
            "referer": self.Purchase.Url,
            "sec-ch-ua": "\" Not A;Brand\";v=\"99\", \"Chromium\";v=\"100\", \"Google Chrome\";v=\"100\"",
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": "\"macOS\"",
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "user-agent": self.User_Agent,
            "x-requested-with": "XMLHttpRequest"
        }
        
        # Add queue token if available
        if self.QueueItHeaderToken:
            headers["x-queueit-queueitem-v1"] = self.QueueItHeaderToken
        
        # Disable redirect in session
        self.Purchase.Session.disable_redirect()
        
        # Build the status URL
        status_url = f"https://{self.Host}/spa-api/queue/{self.Customer_ID}/{self.EventId}/{self.QueueID}/status?cid={self.culture}&l={urllib.parse.quote(self.LayoutName)}&seid={self.Seid}&sets={self.Sets}"
        
        if self.AlternativePath:
            status_url = f"https://{self.Host}{self.AlternativePath}/spa-api/queue/{self.Customer_ID}/{self.EventId}/{self.QueueID}/status?cid={self.culture}&l={urllib.parse.quote(self.LayoutName)}&seid={self.Seid}&sets={self.Sets}"
        
        try:
            response = self.Purchase.Session.post(
                status_url,
                json=payload,
                headers=headers
            )
            
            # Check if we are redirected to the target
            if "redirectUrl" in response.text and "\"isRedirectToTarget\":true" in response.text:
                link = self.respurl
                self.Purchase.NonCookieCheckoutUrl = self.re_find(response.text, r'"redirectUrl":"(.*?)"')
                
                if "axs." in self.Purchase.NonCookieCheckoutUrl:
                    link = self.Purchase.NonCookieCheckoutUrl
                
                # Check for maintenance queue
                if "maintenance" in response.text:
                    self.Purchase.success("Passed maintenance Queue. Going to real Queue.")
                    return True, None
                
                # Track successful checkouts
                self.Purchase.success("Successfully passed Queue!.")
                self.Purchase.track_checkouts()
               
                return True, None
            
            # Normal queue status handling
            if response.status_code == 200:
                # Save queue token if provided
                if "X-Queueit-Queueitem-V1" in response.headers:
                    self.QueueItHeaderToken = response.headers["X-Queueit-Queueitem-V1"]
                
                # Parse the response
                queue_resp = json.loads(response.text)
                
                # Before queue starts case
                if queue_resp.get("PageID") == "before":
                    seconds_to_start = queue_resp.get("Ticket", {}).get("SecondsToStart", 0)
                    self.Purchase.info(f"Queue has not started yet. Waiting for Queue to start in {seconds_to_start} seconds.")
                    
                    if seconds_to_start > 10:
                        self.Purchase.interaction("Sleeping until 10 seconds before Start")
                        time.sleep(seconds_to_start - 10)
                    else:
                        time.sleep(3)
                else:
                    # Get ticket details
                    ticket = queue_resp.get("ticket", {})
                    queue_number = ticket.get("queueNumber")
                    users_ahead = ticket.get("usersInLineAheadOfYou")
                    progress = ticket.get("progress")
                    
                    # Check for error conditions
                    if queue_number is None and users_ahead is None and progress is None:
                        if "softblock" in response.text:
                            self.Purchase.error_with_ip_change("IP-Block. Restarting Task with new proxy..")
                            self.Purchase.Session.delete_all_cookies()
                            self.buy()
                            time.sleep(9 * 60 * 60)
                        elif "afterevent.aspx" in response.text:
                            self.Purchase.info("Event ended! Task terminating.")
                            time.sleep(9 * 60 * 60)
                        else:
                            self.Purchase.error(f"unknown response detected: {response.text}")
                        
                        return False, None
                    
                    # Format queue number for display
                    formatted_queue_number = queue_number
                    if queue_number is None:
                        formatted_queue_number = "-"
                    
                    # Queue paused case
                    if ticket.get("QueuePaused", False):
                        self.Purchase.info(f"[QUEUE PAUSED] Waiting in Queue, Number: {formatted_queue_number}, Users ahead: {users_ahead}, Progress: {progress}")
                    else:
                        self.QueueLiveTries += 1
                        
                        # Mark queue as started after some successful tries
                        if self.QueueLiveTries > 3:
                            # Set global variable indicating queue has started
                            # In original code this sets sharedvariables.HasQueueStarted = true
                            pass
                        
                        self.Purchase.info(f"Waiting in Queue, Number: {formatted_queue_number}, Users ahead: {users_ahead}, Progress: {progress}")
                    
                    # Update product name from queue texts
                    texts = queue_resp.get("Texts", {})
                    self.Purchase.ProductName = texts.get("Header", "")
                    
                    # Wait for next poll
                    self.Purchase.wait_delay_monitor()
                
                return False, None
            elif response.status_code == 429:
                self.Purchase.error("Error waiting in Queue -- Rate Limited -- Changing IP -- Retrying")
                return False, None
            elif response.status_code in [403, 404]:
                self.Tries += 1
                self.Purchase.error_with_ip_change("Error waiting in Queue -- Access Denied, Restarting -- Changing IP -- Retrying")
                
                if self.Tries % 3 == 0:
                    self.Purchase.Session.delete_all_cookies()
                    self.buy()
                    time.sleep(9 * 60 * 60)
                    return False, None
                
                return False, None
            elif response.status_code == 302 and "softblock" in response.text:
                self.Purchase.error_with_ip_change(f"Error waiting in Queue -- SOFTBLOCK [{response.status_code}]")
                self.Purchase.info(response.text)
                self.Purchase.Session.delete_all_cookies()
                self.buy()
                time.sleep(9 * 60 * 60)
                return False, None
            else:
                self.Purchase.error(f"Error waiting in Queue -- Unknown Error [{response.status_code}]")
                return False, None
        except Exception as e:
            return False, e 

    def get_queue(self):
        """
        Make the initial request to a website and check if it uses Queue-it
        
        Returns:
            Tuple containing success status (bool) and optional error
        """
        self.check_if_queue_has_started()
        
        headers = {
            "accept": "*/*",
            "accept-encoding": "gzip, deflate, br",
            "accept-language": "de-DE,de;q=0.9,en-US;q=0.8,en;q=0.7",
            "cache-control": "no-cache",
            "pragma": "no-cache",
            "sec-ch-ua": "\" Not A;Brand\";v=\"99\", \"Chromium\";v=\"100\", \"Google Chrome\";v=\"100\"",
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": "\"macOS\"",
            "sec-fetch-dest": "document",
            "sec-fetch-mode": "navigate",
            "sec-fetch-site": "none",
            "sec-fetch-user": "?1",
            "upgrade-insecure-requests": "1",
            "user-agent": self.User_Agent
        }
        
        try:
            # First, check if the URL already contains queue-it parameters
            if "queue-it.net" in self.Purchase.Url and "c=" in self.Purchase.Url:
                # This is already a Queue-it URL with parameters
                # Extract parameters from the URL
                parsed_url = urllib.parse.urlparse(self.Purchase.Url)
                query_params = urllib.parse.parse_qs(parsed_url.query)
                
                # Set Queue-it parameters
                self.QueueItEnabled = True
                self.respurl = self.Purchase.Url
                self.respstatcode = "200"  # Assume success
                self.Host = parsed_url.netloc
                
                # Extract parameters
                if 'c' in query_params:
                    self.Customer_ID = query_params['c'][0]
                if 'e' in query_params:
                    self.EventId = query_params['e'][0]
                if 'q' in query_params:
                    self.QueueID = query_params['q'][0]
                if 'cid' in query_params:
                    self.culture = query_params['cid'][0]
                if 't' in query_params:
                    self.TargetUrl = urllib.parse.unquote(query_params['t'][0])
                
                # Try to make a request to get the page content
                response = self.Purchase.Session.get(self.Purchase.Url, headers=headers)
                self.resptext = response.text
                
                # Log extracted parameters
                self.Purchase.info(f"Extracted Queue-it parameters from URL:")
                self.Purchase.info(f"Customer ID: {self.Customer_ID}")
                self.Purchase.info(f"Event ID: {self.EventId}")
                self.Purchase.info(f"Queue ID: {self.QueueID}")
                self.Purchase.info(f"Target URL: {self.TargetUrl}")
                
                return True, None
            
            # Otherwise, make a standard request
            response = self.Purchase.Session.get(self.Purchase.Url, headers=headers)
            
            # Check if the URL contains 'c=' which indicates Queue-it is active
            if "c=" in response.url:
                self.resptext = response.text
                
                self.respstatcode = str(response.status_code)
                self.QueueItEnabled = True
                
                # Parse the redirected URL to extract queue parameters
                parsed_url = urllib.parse.urlparse(response.url)
                query_params = urllib.parse.parse_qs(parsed_url.query)
                
                # Extract parameters
                if 'c' in query_params:
                    self.Customer_ID = query_params['c'][0]
                if 'e' in query_params:
                    self.EventId = query_params['e'][0]
                if 'q' in query_params:
                    self.QueueID = query_params['q'][0]
                if 'cid' in query_params:
                    self.culture = query_params['cid'][0]
                if 't' in query_params:
                    self.TargetUrl = urllib.parse.unquote(query_params['t'][0])
                
                self.Host = parsed_url.netloc
                
                # Get Queue-it cookie or user ID
                self.QueueItCookie = self.Purchase.Session.get_cookie("Queue-it", "")
                if not self.QueueItCookie:
                    self.QueueItCookie = self.re_find(response.text, r'data-userid="(.*?)"')
                
                self.respurl = response.url
                
                # Handle maintenance queue
                if "maintenance" in response.url:
                    self.Purchase.interaction("Maintenance Queue!")
                    self.Purchase.interaction("Solving Queue-It")
                    
                    # Solve Queue-It
                    self.solve_queue_it()
                    
                    # Handle Proof of Work challenge if needed
                    if self.pow:
                        self.get_pow_challenge()
                        self.Purchase.interaction("Solving PoW")
                        
                        postfix, err = self.solve_pow_challenge(self.Input, self.Complexity, self.Runs)
                        if err:
                            self.Purchase.error("Error solving POW. Terminating Task.")
                            time.sleep(9 * 60 * 60)  # Sleep for 9 hours
                        
                        solution_data = json.dumps(postfix)
                        self.Solution = solution_data
                        self.Purchase.success("Solved PoW")
                        self.submit_pow_solution()
                    
                    # Get queue ID and poll the queue
                    self.get_queue_id()
                    self.poll_queue()
                    
                    return False, None
                
                # Handle softblock
                if "softblock" in response.url:
                    self.softblock = True
                    self.Purchase.error("Queue entry Softblocked!, solving Entry Captcha -- retrying")
                    
                    # Special case for rolandgarros
                    if "rolandgarros" in response.url:
                        self.Purchase.error("Queue entry Softblocked!, TERMINATING")
                        time.sleep(99 * 60 * 60)  # Sleep for 99 hours
                    
                    # Extract the inqueueUrl for later use
                    inqueue_url = self.re_find(response.text, r"inqueueUrl: '(.*?)'")
                    if inqueue_url:
                        inqueue_url = urllib.parse.unquote(inqueue_url)
                        self.Purchase.info(f"Found inqueue URL: {inqueue_url}")
                        
                        # Solve the captcha challenges
                        self.Purchase.interaction("Solving Queue-It")
                        self.solve_queue_it()
                        
                        # Construct the new UR
                        new_url = f"https://footlocker.queue-it.net/?c=footlocker&e=cxcdtest02&t=https%3A%2F%2Fwww.footlocker.com%2Fsearch%3Fquery%3D6H053P85&cid=de-DE&scv={self.sessioninfo}"
                        self.Purchase.info(f"Redirecting to queue with session token: {new_url}")
                        
                        # Update the URL for the next request
                        self.Purchase.Url = new_url
                        return False, None
                    else:
                        self.Purchase.error("Could not find inqueueUrl in softblock page")
                        return False, Exception("No inqueueUrl found in softblock page")
                else:
                    # Remove first two characters if present (matching Go code pattern)
                    if len(self.QueueItCookie) > 2 and self.QueueItCookie.startswith("g_"):
                        self.QueueItCookie = self.QueueItCookie[2:]
                
                self.softblock = False
                return True, None
            else:
                self.QueueItEnabled = False
                self.Purchase.error_with_ip_change(f"[{response.status_code}] Queue-IT not activated yet. Retrying.")
                self.Purchase.wait_delay_monitor()
                return False, None
                
        except Exception as e:
            return False, e 