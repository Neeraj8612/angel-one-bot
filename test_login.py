from SmartApi import SmartConnect
import pyotp, os
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

api_key = os.getenv("API_KEY")
client_id = os.getenv("CLIENT_ID")
pwd = os.getenv("CLIENT_PWD")
totp = os.getenv("TOTP_SECRET")

print("🔑 Testing login with:")
print("API_KEY:", api_key)
print("CLIENT_ID:", client_id)

# Initialize SmartAPI
obj = SmartConnect(api_key=api_key)

# Generate OTP
otp = pyotp.TOTP(totp).now()
print("Generated OTP:", otp)

# Try login
data = obj.generateSession(client_id, pwd, otp)
print("Login Response:", data)
