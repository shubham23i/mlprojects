import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.mlproject.exception import custom_exception
from src.mlproject.logger import logging
from src.mlproject.utils import save_object
import os
import sys
from dataclasses import dataclass

@dataclass
class DataTransformationConfig():
    preprocessor_obj_file_path=os.path.join('artifacts','preprocesser.pkl')
class DataTransformation():
    def __init__(self):
        self.data_tranformation_config=DataTransformationConfig()

    def get_data_transformer_obj(self):
        """this function is responsible for data transformation"""
        try:
            numerical_columns = ["writing score", "reading score"]
            categorical_columns = [
                "gender",
                "race/ethnicity",
                "parental level of education",
                "lunch",
                "test preparation course",
            ]
            num_pipeline=Pipeline(steps=[
                ("imputer",SimpleImputer(strategy='median')),
                ('scalar',StandardScaler())])
            cat_pipeline=Pipeline(steps=[
            ("imputer",SimpleImputer(strategy='most_frequent')),
            ("onehotencoder",OneHotEncoder()),
            ("scaler",StandardScaler(with_mean=False))
            ])
            logging.info(f"categorical columns {categorical_columns}")
            logging.info(f"numerical columns {numerical_columns}")

            preprocessor=ColumnTransformer([
                ('num_pipeline',num_pipeline,numerical_columns),
                ('cat_pipeline',cat_pipeline,categorical_columns)
            ])
            return preprocessor
        except Exception as e:
            raise custom_exception(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info('reading train and test files')
            preprocessing_obj=self.get_data_transformer_obj()
            target_column="math score"
            numerical_columns=['writing score','reading score']

            input_freatures_train_df=train_df.drop(columns=target_column,axis=1)
            target_freatures_train_df=train_df[target_column]
            input_freatures_test_df=test_df.drop(columns=target_column,axis=1)
            target_freatures_test_df=test_df[target_column]
            logging.info('applying preprocessing on train and test df')

            input_feature_train_arr=preprocessing_obj.fit_transform(input_freatures_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_freatures_test_df)

            train_arr=np.c_[input_feature_train_arr,np.array(target_freatures_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_freatures_test_df)]
            logging.info('saved preprocessing object')


            save_object(
                file_path=self.data_tranformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )
            return (
                train_arr,
                test_arr,
                self.data_tranformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise custom_exception(e,sys)
