# HV-GS Steganography Tool

An adaptive LSB image steganography system using machine learning and key-based randomization.
-> Resistamt to LSBR kind attacks (Tested ✔️)


<img width="1918" height="1025" alt="hvgs_tool" src="https://github.com/user-attachments/assets/8711df78-387e-47b9-acd8-73cab02cecae" />

<img width="1919" height="1018" alt="hvgs_tool_extract" src="https://github.com/user-attachments/assets/c0f36908-f47f-44e3-9333-d7305746552e" />



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
