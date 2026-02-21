import dagshub

dagshub.init(repo_owner='shubham23i', repo_name='mlprojects', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)