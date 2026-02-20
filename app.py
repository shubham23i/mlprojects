from src.mlproject.logger import logging
from src.mlproject.exception import custom_exception
import sys
from src.mlproject.components.data_ingestion import DataIngestion
from src.mlproject.components.data_ingestion import DataIngestionConfig



if __name__=="__main__":
    logging.info("the execution has started")

    try:
        data_ingestion=DataIngestion()
        data_ingestion.initiate_dataingestion()
        #data_ingestion_config=DataIngestionConfig()
        
    except Exception as e:
        logging.info("custom exception")
        raise custom_exception(e,sys)