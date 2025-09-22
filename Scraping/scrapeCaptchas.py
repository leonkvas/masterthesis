import requests
from requests import post
from base64 import b64decode

proxy = "gw-eu.lemonclub.io:5555:pkg-lemonprime-country-US:old6zc33df86s1u8" # rotating Proxy to not get banned while scraping
proxy_ip, proxy_port, proxy_user, proxy_pw = proxy.split(":")
proxy_url = f"http://{proxy_user}:{proxy_pw}@{proxy_ip}:{proxy_port}"
s = requests.Session()
s.proxies = {"http": proxy_url, "https": proxy_url}

def load_captcha():
    url = "https://footlocker.queue-it.net/challengeapi/queueitcaptcha/challenge/en-us"
    headers = {
        "Host": "footlocker.queue-it.net",
        "sec-ch-ua": '"Chromium";v="122", "Not(A:Brand";v="24", "Google Chrome";v="122"',
        "sec-ch-ua-mobile": "?0",
        "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "sec-ch-ua-platform": '"macOS',
        "accept": "*/*",
        "origin": "https://footlocker.queue-it.net",
        "sec-fetch-site": "same-origin",
        "sec-fetch-mode": "cors",
        "sec-fetch-dest": "empty",
        "referer": "https://footlocker.queue-it.net/",
        "accept-language": "en-GB,en;q=0.9",
        "x-queueit-challange-customerid": "footlocker",
        "x-queueit-challange-eventid": "cxcdtest02",
        "x-queueit-challange-hash": "sg7R/eOE9jBw1RMb1iI7d1M8uKgf6sErktuA3Q69hgw=",
        "x-queueit-challange-reason": "1"
    }
    response = s.post(url, headers=headers)
    if response.status_code == 200:
        return response.json().get('imageBase64')
    return None

if __name__ == "__main__":
    for i in range(500):  # Iterate to download n images
        b64image = load_captcha()
        if b64image:
            filename = f"queueit_3_{i+1}.png"
            with open("scrapedCaptchas/"+filename, "wb") as fh:
                fh.write(b64decode(b64image))
            print(f"Saved: {filename}")
        else:
            print(f"Failed to retrieve image {i+1}")
