import os
import sys
from src.mlproject.exception import custom_exception
from src.mlproject.logger import logging
import pandas as pd
from dotenv import load_dotenv
import pymysql
import pickle
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score


# 1. Initialize load_dotenv() at the top
load_dotenv()

def read_sql_data():
    logging.info('Reading MySQL database started')
    try:
        # 2. Get variables INSIDE the function to ensure they are captured correctly
        host = os.getenv('host')
        user = os.getenv('user')
        password = os.getenv('password')
        db = os.getenv('db')

        # 3. Debugging: This helps you see if the values are actually being read
        print(f"Connecting to {host} as {user}...")

        my_db = pymysql.connect(
            host=host,
            user=user,
            password=password,
            db=db
        )
        
        # 4. Fixed the logging call (it only takes one main message argument)
        logging.info(f"Connection established: {my_db}")
        
        df = pd.read_sql_query('select * from students', my_db)
        
        print(df.head())
        
        # 5. Always close your connection!
        my_db.close()
        
        return df

    except Exception as e:
        raise custom_exception(e, sys)
    
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise custom_exception(e, sys)
    
def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report
    except Exception as e:
        raise custom_exception(e, sys)