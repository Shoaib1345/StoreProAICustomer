# Text-to-Speech-

Setup & Run
# 1️⃣ Create virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
# 2️⃣ Install dependencies
```
pip install flask
```
```
pip install sentence-transformers
```


``` 
pip install numpy
```
``` 
pip install scikit-learn
```
``` 
pip install datasets
```
``` 
pip install transformers[torch]
```
```
python train.py
```
# 3️⃣ Run the Flask app
python app.py


Then open in browser:
👉 http://127.0.0.1:5000

🧠 Features

# 📁 Folder Structure
```
TexttoSpeech/
│
├── app.py
├── templates/
│   └── about.html
|   └── index.html
└── static/
    -images
    ├── javascript/
    │   └── index.js
    ├── css/
        └── about.css
        └── index.css
    
```
