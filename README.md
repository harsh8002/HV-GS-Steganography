# HV-GS Steganography Tool

An adaptive LSB image steganography system using machine learning and key-based randomization.
-> Resistamt to LSBR kind attacks (Tested ✔️)

## Features

* ML-based channel selection (LightGBM + Decision Tree)
* Key-controlled pixel randomization
* LSB embedding
* GUI-based tool (CustomTkinter)

## Requirements

* Python 3.8+
* Install dependencies:

```
pip install -r requirements.txt
```

## How to Run

```
Keep both the models channel_selecter_model.pkl and channel_model.pkl in same folder as main.py and run main.py

python main.py
```

## Usage

1. Select an image
2. Enter secret key
3. Choose model
4. Hide or Extract data

## Notes

* Keep both `.pkl` model files in the same folder
* Same key must be used for extraction

## Author

Harsh Vardhana, Dr. Gaurav sundaram
