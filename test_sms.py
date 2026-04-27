# test_sms.py
import requests

API_KEY      = "uqmFAfh6vIwesKY2rbj5oDV83Hyp1a4NiSTLgB79GRCZnQcOMWtuM1PDQk82vxoVUnjHTmbLIKE7SafG"  # ← your Fast2SMS key
PHONE_NUMBER = "8220475476"                # ← YOUR own mobile number

message = "TEST: Safety monitoring SMS working!"

url = "https://www.fast2sms.com/dev/bulkV2"
payload = {
    "route":    "q",
    "message":  message,
    "language": "english",
    "flash":    0,
    "numbers":  PHONE_NUMBER,
}
headers = {
    "authorization": API_KEY,
    "Content-Type":  "application/x-www-form-urlencoded",
}

response = requests.post(url, data=payload, headers=headers, timeout=10)
print("Status:", response.status_code)
print("Response:", response.json())