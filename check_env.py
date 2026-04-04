from dotenv import load_dotenv
import os

load_dotenv()

key = os.getenv("GEMINI_API_KEY")

if key:
    print(f"✅ API Key found: {key[:10]}...")
else:
    print("❌ API Key NOT found — check your .env file")
    print("\nFiles in current folder:")
    for f in os.listdir("."):
        print(f"  {f}")