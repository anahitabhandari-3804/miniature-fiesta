import subprocess
import sys

# Function to install packages
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Read the requirements file
with open("C:\Users\DELL\Desktop\college\research or project\DL lab\Face-Recognition-Attendance-Projects-main\Face-Recognition-Attendance-Projects-main\requirements.txt") as f:
    for line in f:
        package = line.strip()
        try:
            # Try importing the package to check if it's already installed
            __import__(package.split("==")[0])
            print(f"{package} is already installed.")
        except ImportError:
            # Install the package if it's not installed
            print(f"Installing {package}...")
            install(package)
