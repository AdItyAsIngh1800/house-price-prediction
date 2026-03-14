# **🏠 House Price Prediction System**

  

Machine Learning & Full-Stack Deployment Project

![Python](https://img.shields.io/badge/Python-3.10-blue)

![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)

![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)

![Docker](https://img.shields.io/badge/Docker-Container-blue)

![License](https://img.shields.io/badge/License-MIT-yellow)

Author: **Aditya Singh**

----------

# **📌 Project Overview**

  

This project is a  **full-stack machine learning system**  designed to predict house prices using historical property transaction data.

  

The system integrates:

-   Machine Learning model training
    
-   Backend API for predictions
    
-   Interactive web dashboard
    
-   Database logging of predictions
    
-   Containerized deployment using Docker
    

  

The project demonstrates an **end-to-end ML workflow from data preprocessing to deployment.**

----------

# **✨ Features**

  

✔ Data preprocessing pipeline

✔ Categorical feature encoding

✔ Multiple regression models trained

✔ Model performance comparison

✔ Linear Regression model used for deployment

✔ REST API built with FastAPI

✔ Interactive dashboard using Streamlit

✔ Prediction logging to SQLite database

✔ Docker containerization for reproducible deployment

----------

# **⚙️ Tech Stack**

  

## **Programming Language**

-   Python
    

  

## **Machine Learning**

-   Scikit-learn
    
-   Pandas
    
-   NumPy
    

  

## **Backend API**

-   FastAPI
    
-   Uvicorn
    

  

## **Frontend Dashboard**

-   Streamlit
    

  

## **Database**

-   SQLite
    

  

## **Deployment**

-   Docker
    
-   Docker Compose
    

----------

# **🧠 Machine Learning Pipeline**

  

The machine learning workflow followed in this project:

1.  Load historical property transaction dataset
    
2.  Clean and preprocess dataset
    
3.  Encode categorical variables
    
4.  Train regression models
    
5.  Evaluate model performance
    
6.  Select model for deployment
    
7.  Deploy prediction API using FastAPI
    
8.  Build interactive UI using Streamlit
    
9.  Log predictions to SQLite database
    
10.  Containerize the system using Docker
    

----------

# **📊 Dataset**

  

Dataset used:

  

👉 https://www.kaggle.com/datasets/hm-land-registry/uk-housing-prices-paid

  

The dataset contains historical UK property transaction records including:

-   property type
    
-   location (town/city, district, county)
    
-   property condition (new/old)
    
-   ownership duration
    
-   transaction date
    
-   sale price
    

  

The dataset is published by the  **UK Land Registry**  and contains large-scale real estate transaction data.

----------

## **📊 Model Performance**

  Several regression models were trained and evaluated using MAE, MSE, RMSE, R² Score, and cross-validation.

## 📊 Model Performance Comparison

| Model | MAE | MSE | RMSE | R2 Score | CV Mean R2 |
|------|------|------|------|------|------|
| Linear Regression | 84793.5665 | 35531128923 | 188497.0263 | 0.2360559754 | 0.1205277606 |
| Decision Tree | 66668.81589 | 78385657903 | 279974.3879 | -0.6853462522 | -0.04580773359 |
| Gradient Boosting | 69202.06955 | 84768597990 | 291150.4731 | -0.8225839107 | -0.1003212652 |
| Random Forest Tuned | 64146.22748 | 57402220388 | 239587.6048 | -0.2341877275 | 0.07344626178 |

**Linear Regression achieved the best overall performance and was selected as the final deployed model.**

  

Although tree-based models produced lower MAE in some cases, their negative R² scores indicate weaker overall fit and poor generalization on this dataset.

  

The dataset mainly contains transaction metadata such as:

-   property type
    
-   ownership duration
    
-   location (town, district, county)
    
-   transaction year
    

  

but lacks important property attributes like:

-   bedrooms
    
-   floor area
    
-   bathrooms
    
-   property condition
    

  

Because of this, simpler linear relationships performed better than more complex non-linear models.

----------

## **⚙️ Model Optimization**

  

Hyperparameter tuning was performed using  **RandomizedSearchCV**  for the Random Forest model.

  

This automated search evaluated multiple parameter combinations including:

-   number of trees
    
-   maximum depth
    
-   minimum samples per split
    
-   minimum samples per leaf
    

  

Although tuning improved model stability, Linear Regression still produced the highest R² score and was therefore selected for deployment.

----------

## 🧪 Experiment Tracking

Model training experiments are tracked using **MLflow**.

MLflow records:

- model parameters
- evaluation metrics
- cross-validation scores
- trained model artifacts

Each training run is stored as an experiment, allowing easy comparison between multiple models such as:

- Linear Regression
- Decision Tree
- Gradient Boosting
- Random Forest

This makes the training pipeline reproducible and improves experiment management.

---

# **🏗 System Architecture**

```mermaid
flowchart LR

subgraph Frontend
U[User]
S[Streamlit App]
end

subgraph Backend
A[FastAPI API]
end

subgraph ML
P[ML Pipeline<br/>Encoder + Model]
end

subgraph Storage
D[(SQLite Database)]
end

subgraph Training
C[CSV Dataset]
P2[Preprocessing]
T[Training & Evaluation]
end

U --> S
S -->|Send input| A
A --> P
A -->|Log prediction| D
A -->|Return predicted price| S

C --> P2 --> T --> P
```

----------

# **📂 Project Structure**

```
house-price-prediction/
├── Dockerfile
├── LICENSE
├── README.md
├── Screenshots
│   ├── fastapi_docs.png
│   └── streamlit_ui.png
├── api
│   ├── __pycache__
│   │   ├── main.cpython-311.pyc
│   │   ├── main.cpython-314.pyc
│   │   ├── schemas.cpython-311.pyc
│   │   └── schemas.cpython-314.pyc
│   ├── main.py
│   └── schemas.py
├── app
│   └── streamlit_app.py
├── data
│   ├── dataset_sample.md
│   ├── extract_metadata.ipynb
│   └── price_paid_records.csv
├── database
│   ├── __pycache__
│   │   ├── db.cpython-311.pyc
│   │   └── db.cpython-314.pyc
│   └── db.py
├── docker-compose.yml
├── house_predictions.db
├── models
│   ├── county_encoder.pkl
│   ├── district_encoder.pkl
│   ├── duration_encoder.pkl
│   ├── house_price_model.pkl
│   ├── house_price_pipeline.pkl
│   ├── model_metrics.csv
│   ├── old_new_encoder.pkl
│   ├── property_type_encoder.pkl
│   └── town_city_encoder.pkl
├── notebooks
│   └── house_price_prediction.ipynb
├── requirements.txt
└── src
    ├── __pycache__
    │   └── data_preprocessing.cpython-314.pyc
    ├── data_preprocessing.py
    ├── predict.py
    └── train_model.py
```

----------

# **📸 Application Preview**

  

## **Streamlit Interface**
![Streamlit UI](Screenshots/streamlit_ui.png)

## **FastAPI Swagger Documentation**
![FastAPI Docs](Screenshots/fastapi_docs.png)

----------

# **🚀 Running the Project Locally**

  

## **1️⃣ Clone repository**

```
git clone https://github.com/AdItyAsIngh1800/house-price-prediction.git
cd house-price-prediction
```

2️⃣ Install dependencies

```
pip install -r requirements.txt
```

----------

3️⃣ Start FastAPI server

```
uvicorn api.main:app --reload
```

FastAPI documentation available at:

```
http://localhost:8000/docs
```

----------

4️⃣ Run Streamlit app

```
streamlit run app/streamlit_app.py
```

Streamlit runs at:

```
http://localhost:8501
```

----------

🐳 Running with Docker

  

Build containers:

```
docker compose build
```

Start containers:

```
docker compose up
```

This will start:

  

• FastAPI service

• Streamlit dashboard

• SQLite database logging

----------

🔌 API Endpoint

  

POST /predict

  

Predict house price.

  

Example request:

```
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d '{
"property_type": "D",
"old_new": "Y",
"duration": "F",
"town_city": "LONDON",
"district": "CAMDEN",
"county": "GREATER LONDON",
"year": 2015
}'
```

Example response:

```
{
"predicted_price": 452000
}
```

----------

🗄 Prediction Logging

  

Each prediction is stored in a SQLite database including:

  

• property type

• location details

• transaction year

• predicted price

• timestamp

  

This enables tracking of prediction history and usage patterns.

----------

⚠ Limitations

  

The dataset does not include several important real estate features such as:

  

• number of bedrooms

• property area

• number of bathrooms

• building condition

  

Therefore predictions are based mainly on:

  

• property type

• ownership duration

• location

• transaction year

----------

🚀 Future Improvements

  

Possible improvements include:

  

• Adding additional property features

• Implementing advanced models (XGBoost, LightGBM)

• Adding geospatial location features

• Deploying the system on cloud platforms (AWS / GCP / Azure)

• Adding authentication and analytics dashboard

----------

💼 CV Project Description

  

Built and deployed a full-stack machine learning system to predict property prices using historical UK Land Registry data. Implemented data preprocessing, feature engineering, and regression model comparison with Scikit-learn. Deployed the model through a FastAPI REST API with a Streamlit interactive dashboard, integrated SQLite database logging, and containerized the application using Docker.

----------

📜 License

  

This project is licensed under the MIT License.

----------
