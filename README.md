# A Simple DeepWalk implementation using PyTorch

This implementation uses CBOW instead of Skip-Gram

## Prerequisites
Ensure you have the following installed on your system:

- Python 3.x
- pip

## How To Run

Open a terminal in the project's directory and run the following commands:

### Linux/MacOS
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python DeepWalk.py
```

### Windows

```
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
python DeepWalk.py
```

## During Runtime

The script gives you the ability to add custom parameters if you so please, you also have the option to run with default parameters.
Once the DeepWalk algorithm has been trained the results will be displayed in a scatter graph.
