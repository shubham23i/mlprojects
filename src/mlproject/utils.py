import os
import sys
from src.mlproject.exception import custom_exception
from src.mlproject.logger import logging
import pandas as pd
from dotenv import load_dotenv
import pymysql

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