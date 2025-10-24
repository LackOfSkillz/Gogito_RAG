import subprocess, sys, os

print("🔧 Checking & Installing Cogito RAG dependencies...\n")

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package])

req_file = os.path.join(os.path.dirname(__file__), "requirements.txt")

try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_file])
    print("\n✅ All dependencies installed successfully.")
except Exception as e:
    print(f"❌ Error installing dependencies: {e}")
