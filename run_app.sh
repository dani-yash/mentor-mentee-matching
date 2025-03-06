#!/bin/bash

echo "Setting up the Mentor-Mentee Matching Tool..."

# 1. Check if Python3 is installed
if ! command -v python3 &> /dev/null
then
    echo "Python3 is not installed. Attempting to install Python3..."
    if command -v brew &> /dev/null; then
        brew install python3
    else
        echo "Homebrew is not installed. Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        
        # Add Homebrew to PATH for immediate usage
        eval "$(/opt/homebrew/bin/brew shellenv)"
        
        echo "Homebrew installed. Installing Python3..."
        brew install python3
    fi
fi

# 2. Check if pip is installed
if ! command -v pip3 &> /dev/null
then
    echo "pip3 is not installed. Installing pip..."
    python3 -m ensurepip --upgrade
fi

# 3. Create and activate a virtual environment
if [ ! -d "venv" ]; then
    echo "Creating a virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "Installing dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
else
    echo "Activating the existing virtual environment..."
    source venv/bin/activate
fi

# 4. Run the Streamlit app
echo "Starting the Streamlit app..."
streamlit run app.py

# 5. Keep terminal open in case of errors
echo "Press [CTRL+C] to stop the app."
wait