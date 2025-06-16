import time
import requests
import os
import random
import threading
from queue_it_purchase import QueueItPurchase
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
import queue

# Add a queue for thread-safe printing
print_queue = queue.Queue()

class Purchase:
    """
    Simple Purchase class to simulate the main functionality needed by QueueItPurchase
    """
    def __init__(self, url, user_agent=None, proxy=None, task_nr=1):
        self.Url = url
        self.Proxy = proxy
        self.ProxyList = []
        self.CurrentProxyIndex = 0
        self.TaskNr = task_nr  # Set the task number
        self.load_proxies()
        
        # Initialize with the first proxy if available
        initial_proxy = self.get_next_proxy() if self.ProxyList else proxy
        self.Session = SessionWrapper(proxy=initial_proxy, task_nr=self.TaskNr)
        
        self.User_Agent = user_agent or "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.75 Safari/537.36"
        self.ProductName = ""
        self.Image = ""
        self.Size = ""
        self.Site = ""
        self.CurrentCsvLine = {"SIZE": "1000"}  # Default size limit
        self.RealUrl = url
        self.NonCookieCheckoutUrl = ""
        self.Debug = False  # Set to False to disable saving captcha images
    
    def load_proxies(self):
        """Load proxies from proxies.txt file"""
        try:
            proxy_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "proxies.txt")
            
            if os.path.exists(proxy_file):
                with open(proxy_file, 'r') as f:
                    self.ProxyList = [line.strip() for line in f if line.strip()]
                
                if self.ProxyList:
                    print_queue.put(f"[TASK {self.TaskNr}] [PROXY] Loaded {len(self.ProxyList)} proxies from proxies.txt")
                else:
                    print_queue.put(f"[TASK {self.TaskNr}] [PROXY] No proxies found in proxies.txt")
            else:
                print_queue.put(f"[TASK {self.TaskNr}] [PROXY] proxies.txt file not found. Running without proxies.")
        except Exception as e:
            print_queue.put(f"[TASK {self.TaskNr}] [PROXY] Error loading proxies: {str(e)}")
    
    def get_next_proxy(self):
        """Get the next proxy from the list"""
        if not self.ProxyList:
            return None
            
        proxy = self.ProxyList[self.CurrentProxyIndex]
        self.CurrentProxyIndex = (self.CurrentProxyIndex + 1) % len(self.ProxyList)
        return proxy
    
    def info(self, message):
        """Print info message"""
        print_queue.put(f"[TASK {self.TaskNr}] [INFO] {message}")
    
    def error(self, message):
        """Print error message"""
        print_queue.put(f"[TASK {self.TaskNr}] [ERROR] {message}")
        # Add sleep after any error
        time.sleep(3)
    
    def error_with_ip_change(self, message):
        """Print error message and signal IP change"""
        print_queue.put(f"[TASK {self.TaskNr}] [ERROR] {message} - IP change needed")
        # Add sleep before rotating proxy
        time.sleep(3)
        self.rotate_proxy()
    
    def success(self, message):
        """Print success message"""
        print_queue.put(f"[TASK {self.TaskNr}] [SUCCESS] {message}")
    
    def interaction(self, message):
        """Print interaction message"""
        print_queue.put(f"[TASK {self.TaskNr}] [INTERACTION] {message}")
    
    def debug(self, message):
        """Print debug message"""
        if self.Debug:
            print_queue.put(f"[TASK {self.TaskNr}] [DEBUG] {message}")
    
    def rotate_proxy(self):
        """Rotate proxy and update session"""
        new_proxy = self.get_next_proxy()
        if new_proxy:
            print_queue.put(f"[TASK {self.TaskNr}] [PROXY] Rotating proxy to: {new_proxy}")
            # Create a new session with the new proxy
            self.Session = SessionWrapper(proxy=new_proxy, task_nr=self.TaskNr)
            self.Proxy = new_proxy
            # Add sleep after proxy rotation to allow connection to stabilize
            time.sleep(3)
            return True
        else:
            print_queue.put(f"[TASK {self.TaskNr}] [PROXY] No proxies available for rotation")
            # Sleep even when proxy rotation fails
            time.sleep(3)
            return False
    
    def extension_checkout(self, link, cookies, _, __):
        """Simulate extension checkout"""
        print_queue.put(f"[TASK {self.TaskNr}] [CHECKOUT] Proceeding to checkout at: {link}")
        print_queue.put(f"[TASK {self.TaskNr}] [CHECKOUT] With cookies: {cookies}")
        return None
    
    def track_checkouts(self):
        """Simulate tracking checkouts"""
        print_queue.put(f"[TASK {self.TaskNr}] [SYSTEM] Tracking successful checkout")
    
    def wait_delay_monitor(self):
        """Simulate waiting for next poll"""
        print_queue.put(f"[TASK {self.TaskNr}] [SYSTEM] Waiting for next queue poll...")
        time.sleep(5)

class SessionWrapper:
    """
    Simple wrapper around requests.Session to match the expected interface
    """
    def __init__(self, proxy=None, task_nr=1):
        self.session = requests.Session()
        self.Client = self
        self.jar = self
        self.task_nr = task_nr
        
        # Configure proxy if provided
        if proxy:
            try:
                print_queue.put(f"[TASK {self.task_nr}] [PROXY] Setting up proxy: {proxy}")
                
                # Check if proxy contains username/password or needs them added
                if '@' in proxy:
                    # Format may be: username:password@host:port or username@host:port
                    if ':' in proxy.split('@')[0]:
                        # Already has username:password format
                        print_queue.put(f"[TASK {self.task_nr}] [PROXY] Using provided username:password authentication")
                        proxy_formatted = f"http://{proxy}"
                    else:
                        # Just has username@host:port, need to add password
                        auth_part, host_part = proxy.split('@', 1)
                        proxy_formatted = f"http://{auth_part}:password@{host_part}"
                        print_queue.put(f"[TASK {self.task_nr}] [PROXY] Adding default password to proxy authentication")
                else:
                    # No authentication in URL, may need to be added separately
                    print_queue.put(f"[TASK {self.task_nr}] [PROXY] No authentication in proxy URL, using proxy as is")
                    proxy_formatted = f"http://{proxy}"
                
                print_queue.put(f"[TASK {self.task_nr}] [PROXY] Formatted proxy URL: {proxy_formatted.replace(':password@', ':***@')}")
                
                # Set the proxy for both HTTP and HTTPS
                self.session.proxies = {
                    "http": proxy_formatted,
                    "https": proxy_formatted
                }
                
                # Test the proxy connection
                print_queue.put(f"[TASK {self.task_nr}] [PROXY] Testing proxy connection...")
                test_response = self.session.get("https://httpbin.org/ip", timeout=10)
                if test_response.status_code == 200:
                    print_queue.put(f"[TASK {self.task_nr}] [PROXY] Proxy test successful: {test_response.json()}")
                else:
                    print_queue.put(f"[TASK {self.task_nr}] [PROXY] Proxy test returned status code: {test_response.status_code}")
                    
            except requests.exceptions.ProxyError as pe:
                print_queue.put(f"[TASK {self.task_nr}] [PROXY] Proxy configuration error: {pe}")
                if "407" in str(pe):
                    print_queue.put(f"[TASK {self.task_nr}] [PROXY] 407 Proxy Authentication Required - Please check your proxy credentials")

                # Don't set proxies if there was an error
                self.session.proxies = {}
                
            except Exception as e:
                print_queue.put(f"[TASK {self.task_nr}] [PROXY] Error configuring proxy: {str(e)}")
                self.session.proxies = {}
    
    def post(self, url, headers=None, json=None, data=None, allow_redirects=True):
        """Make a POST request"""
        try:
            response = self.session.post(url, headers=headers, json=json, data=data, allow_redirects=allow_redirects)
            
            # Check for 407 Proxy Authentication Required error
            if response.status_code == 407:
                print_queue.put(f"[TASK {self.task_nr}] [ERROR] 407 Proxy Authentication Required - Check your proxy credentials")
                # Add sleep after proxy auth error
                time.sleep(3)
            
            # Add sleep after any error status code
            if response.status_code >= 400:
                print_queue.put(f"[TASK {self.task_nr}] [ERROR] Request failed with status code: {response.status_code}")
                time.sleep(3)
            
            # Create a custom response object with the properties we need
            class CustomResponse:
                def __init__(self, resp):
                    self.original = resp
                    self.text = resp.text
                    self.headers = resp.headers
                    self.status_code = resp.status_code
                    self.R = self
                    self.url = resp.url
                
                def __getattr__(self, name):
                    return getattr(self.original, name)
            
            return CustomResponse(response)
        except requests.exceptions.ProxyError as pe:
            print_queue.put(f"[TASK {self.task_nr}] [ERROR] Proxy error: {pe}")
            print_queue.put(f"[TASK {self.task_nr}] [INFO] This could be due to incorrect proxy credentials or connectivity issues")
            # Add sleep after proxy error
            time.sleep(3)
            
            # Create a dummy response with 407 status
            class DummyResponse:
                def __init__(self):
                    self.text = "Proxy Authentication Required"
                    self.headers = {}
                    self.status_code = 407
                    self.url = url
            
            class CustomResponse:
                def __init__(self, resp):
                    self.original = resp
                    self.text = resp.text
                    self.headers = resp.headers
                    self.status_code = resp.status_code
                    self.R = self
                    self.url = resp.url
                
                def __getattr__(self, name):
                    return getattr(self.original, name)
            
            return CustomResponse(DummyResponse())
        except Exception as e:
            print_queue.put(f"[TASK {self.task_nr}] [ERROR] Request error: {e}")
            # Add sleep after any exception
            time.sleep(3)
            
            # Create a dummy response with 500 status
            class DummyResponse:
                def __init__(self):
                    self.text = f"Error: {str(e)}"
                    self.headers = {}
                    self.status_code = 500
                    self.url = url
            
            class CustomResponse:
                def __init__(self, resp):
                    self.original = resp
                    self.text = resp.text
                    self.headers = resp.headers
                    self.status_code = resp.status_code
                    self.R = self
                    self.url = resp.url
                
                def __getattr__(self, name):
                    return getattr(self.original, name)
            
            return CustomResponse(DummyResponse())
    
    def get(self, url, headers=None, allow_redirects=True):
        """Make a GET request"""
        try:
            response = self.session.get(url, headers=headers, allow_redirects=allow_redirects)
            
            # Check for 407 Proxy Authentication Required error
            if response.status_code == 407:
                print_queue.put(f"[TASK {self.task_nr}] [ERROR] 407 Proxy Authentication Required - Check your proxy credentials")
                print_queue.put(f"[TASK {self.task_nr}] [INFO] For Panaio proxies, make sure to use the correct password (usually 'password')")
                # Add sleep after proxy auth error
                time.sleep(3)
            
            # Add sleep after any error status code
            if response.status_code >= 400:
                print_queue.put(f"[TASK {self.task_nr}] [ERROR] Request failed with status code: {response.status_code}")
                time.sleep(3)
            
            # Create a custom response object with the properties we need
            class CustomResponse:
                def __init__(self, resp):
                    self.original = resp
                    self.text = resp.text
                    self.headers = resp.headers
                    self.status_code = resp.status_code
                    self.R = self
                    self.url = resp.url
                
                def __getattr__(self, name):
                    return getattr(self.original, name)
            
            return CustomResponse(response)
        except requests.exceptions.ProxyError as pe:
            print_queue.put(f"[TASK {self.task_nr}] [ERROR] Proxy error: {pe}")
            print_queue.put(f"[TASK {self.task_nr}] [INFO] This could be due to incorrect proxy credentials or connectivity issues")
            # Add sleep after proxy error
            time.sleep(3)
            
            # Create a dummy response with 407 status
            class DummyResponse:
                def __init__(self):
                    self.text = "Proxy Authentication Required"
                    self.headers = {}
                    self.status_code = 407
                    self.url = url
            
            class CustomResponse:
                def __init__(self, resp):
                    self.original = resp
                    self.text = resp.text
                    self.headers = resp.headers
                    self.status_code = resp.status_code
                    self.R = self
                    self.url = resp.url
                
                def __getattr__(self, name):
                    return getattr(self.original, name)
            
            return CustomResponse(DummyResponse())
        except Exception as e:
            print_queue.put(f"[TASK {self.task_nr}] [ERROR] Request error: {e}")
            # Add sleep after any exception
            time.sleep(3)
            
            # Create a dummy response with 500 status
            class DummyResponse:
                def __init__(self):
                    self.text = f"Error: {str(e)}"
                    self.headers = {}
                    self.status_code = 500
                    self.url = url
            
            class CustomResponse:
                def __init__(self, resp):
                    self.original = resp
                    self.text = resp.text
                    self.headers = resp.headers
                    self.status_code = resp.status_code
                    self.R = self
                    self.url = resp.url
                
                def __getattr__(self, name):
                    return getattr(self.original, name)
            
            return CustomResponse(DummyResponse())
    
    def delete_all_cookies(self):
        """Delete all cookies"""
        self.session.cookies.clear()
    
    def disable_redirect(self):
        """Disable redirects"""
        self.session.allow_redirects = False
    
    def cookies(self, url):
        """Get cookies for URL"""
        return [{"name": name, "value": value, "domain": domain} 
                for domain, name, value in 
                [(cookie.domain, cookie.name, cookie.value) 
                 for cookie in self.session.cookies]]
    
    def get_cookie(self, name, default=""):
        """Get a cookie by name"""
        for cookie in self.session.cookies:
            if cookie.name == name:
                return cookie.value
        return default

def create_empty_proxies_file():
    """Create an empty proxies.txt file if it doesn't exist"""
    proxy_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "proxies.txt")
    
    if not os.path.exists(proxy_file):
        try:
            with open(proxy_file, 'w') as f:
                f.write("# Add your proxies here, one per line in the format:\n")
                f.write("# pkg-packet-country-US@proxies.panaio.com:8603\n")
            print("[PROXY] Created empty proxies.txt file")
        except Exception as e:
            print(f"[PROXY] Error creating proxies.txt: {str(e)}")

def print_worker():
    """Worker function to handle thread-safe printing"""
    while True:
        message = print_queue.get()
        if message is None:
            break
        print(message)
        print_queue.task_done()

def run_queue_task(task_id, target_url):
    """
    Run a single queue task
    
    Args:
        task_id: Unique identifier for this task
        target_url: The Queue-it URL to process
    """
    try:
        # Create a Purchase instance with the task number
        purchase = Purchase(target_url, task_nr=task_id)
        
        # Set site property for checkout handling
        purchase.Site = "footlocker"
        
        # Create a QueueItPurchase instance
        queue_it = QueueItPurchase(purchase)
        
        # Set the customer ID and event ID for proper detection
        queue_it.Customer_ID = "footlocker"
        queue_it.EventId = "cxcdtest02"
        
        # Call buy instead of get_queue to handle softblocks properly
        print_queue.put(f"[TASK {task_id}] Making initial request and handling any softblocks...")
        success, error = queue_it.buy()
        
        if error:
            print_queue.put(f"[TASK {task_id}] Error in buy process: {error}")
            return
        
        print_queue.put(f"[TASK {task_id}] Successfully connected to Queue-it")
        print_queue.put(f"[TASK {task_id}] Customer ID: {queue_it.Customer_ID}")
        print_queue.put(f"[TASK {task_id}] Event ID: {queue_it.EventId}")
        
        # Check if we're already in a queue or if the queue has been solved
        if queue_it.queueitsolved:
            print_queue.put(f"[TASK {task_id}] Queue-it has been solved! Proceeding to target site.")
            if queue_it.TargetUrl:
                print_queue.put(f"[TASK {task_id}] Proceeding to: {queue_it.TargetUrl}")
        elif queue_it.QueueID:
            print_queue.put(f"[TASK {task_id}] Queue ID: {queue_it.QueueID}")
            
            # Poll the queue until we're through
            print_queue.put(f"[TASK {task_id}] Starting to poll the queue...")
            while True:
                success, error = queue_it.poll_queue()
                
                if error:
                    print_queue.put(f"[TASK {task_id}] Error polling queue: {error}")
                    break
                
                if success:
                    print_queue.put(f"[TASK {task_id}] Successfully passed the queue!")
                    if queue_it.TargetUrl:
                        print_queue.put(f"[TASK {task_id}] Proceeding to: {queue_it.TargetUrl}")
                    break
                    
                # Add a short sleep to avoid hammering the server
                time.sleep(3)
        else:
            print_queue.put(f"[TASK {task_id}] Not in a queue currently. Check the status.")
            
        print_queue.put(f"[TASK {task_id}] Queue-it automation completed.")
        
    except Exception as e:
        print_queue.put(f"[TASK {task_id}] Unexpected error: {str(e)}")

def main():
    """
    Main function to demonstrate Queue-it automation with multiple threads
    """
    # Ensure proxies.txt exists
    create_empty_proxies_file()
    
    # Footlocker Queue-it URL
    target_url = "https://footlocker.queue-it.net/?c=footlocker&e=cxcdtest02"
    
    print(f"Starting Queue-it automation with multiple threads for: {target_url}")
    
    # Start the print worker thread
    print_thread = threading.Thread(target=print_worker, daemon=True)
    print_thread.start()
    
    # Number of concurrent tasks
    num_tasks = 1000
    
    # Create a thread pool
    with ThreadPoolExecutor(max_workers=num_tasks) as executor:
        # Submit tasks
        futures = []
        for i in range(num_tasks):
            task_id = i + 1
            futures.append(executor.submit(run_queue_task, task_id, target_url))
        
        # Wait for all tasks to complete
        for future in futures:
            future.result()
    
    # Signal the print worker to stop
    print_queue.put(None)
    print_thread.join()
    
    print("All tasks completed.")

if __name__ == "__main__":
    main() 