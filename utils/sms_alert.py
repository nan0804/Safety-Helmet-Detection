# utils/sms_alert.py

import os
from dotenv import load_dotenv
from twilio.rest import Client

# Load environment variables
load_dotenv()

ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER")

client = Client(ACCOUNT_SID, AUTH_TOKEN)

def send_sms_alert(phone_number, worker_name, violation_type, timestamp):
    message_body = (
        f"🚨 *SAFETY ALERT* 🚨\n\n"
        f"Hello *{worker_name}*,\n"
        f"You were found violating safety rules.\n\n"
        f"*Violation:* {violation_type}\n"
        f"*Time:* {timestamp}\n\n"
        f"Please wear your PPE immediately!\n"
        f"— Construction Safety System"
    )

    try:
        message = client.messages.create(
            body=message_body,
            from_=FROM_NUMBER,
            to=f"whatsapp:+91{phone_number}"
        )
        print(f"[WhatsApp] Alert sent to {phone_number} ✅ SID: {message.sid}")
        return True
    except Exception as e:
        print(f"[WhatsApp] Error sending message: {e}")
        return False
        