#Service for predicting diabetes based on [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database?resource=download)

## Install deps via venv
1. python3 -m venv venv
2. source venv/bin/activate
3. pip install -r requirements.txt

### Run tests
1. pytest -q -s .

### Train new model 
1. python diabetes_model/train_pipeline.py  

### Run FastAPI app
1. python app/app.py

## Build Docker image
1. docker build -t predict_diabetes .
2. sudo docker run --rm -p 8000:8000 predict_diabetes:latest