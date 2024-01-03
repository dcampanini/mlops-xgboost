import time
from datetime import datetime
from typing import NamedTuple

import kfp
from kfp import dsl
from kfp.v2 import compiler
from kfp.v2.dsl import (Artifact, Dataset, Input, InputPath, Model, Output,
                        OutputPath, ClassificationMetrics, Metrics, component, pipeline)
from kfp.v2.google.client import AIPlatformClient

from google.cloud import aiplatform
from google_cloud_pipeline_components import aiplatform as gcc_aip

PROJECT_ID = 'dev-dlk-cl'
DATASET_ID = "census"  # The Data Set ID where the view sits
VIEW_NAME = "census_data"  # BigQuery view you create for input data

@component(
    packages_to_install=["google-cloud-bigquery==3.10.0"],
)
def create_census_view(
    project_id: str,
    dataset_id: str,
    view_name: str,
):
    """Creates a BigQuery view on `bigquery-public-data.ml_datasets.census_adult_income`.

    Args:
        project_id: The Project ID.
        dataset_id: The BigQuery Dataset ID. Must be pre-created in the project.
        view_name: The BigQuery view name.
    """
    from google.cloud import bigquery

    client = bigquery.Client(project=project_id)

    create_or_replace_view = """
        CREATE OR REPLACE VIEW
        `{dataset_id}`.`{view_name}` AS
        SELECT
          age,
          workclass,
          education,
          education_num,
          marital_status,
          occupation,
          relationship,
          race,
          sex,
          capital_gain,
          capital_loss,
          hours_per_week,
          native_country,
          income_bracket,
        FROM
          `bigquery-public-data.ml_datasets.census_adult_income`
    """.format(
        dataset_id=dataset_id, view_name=view_name
    )

    job_config = bigquery.QueryJobConfig()
    query_job = client.query(query=create_or_replace_view, job_config=job_config)
    query_job.result()
    
    
@component(
    packages_to_install=["google-cloud-bigquery[pandas]==3.10.0"],
)
def export_dataset(
    project_id: str,
    dataset_id: str,
    view_name: str,
    dataset: Output[Dataset],
):
    """Exports from BigQuery to a CSV file.

    Args:
        project_id: The Project ID.
        dataset_id: The BigQuery Dataset ID. Must be pre-created in the project.
        view_name: The BigQuery view name.

    Returns:
        dataset: The Dataset artifact with exported CSV file.
    """
    from google.cloud import bigquery

    client = bigquery.Client(project=project_id)

    table_name = f"{project_id}.{dataset_id}.{view_name}"
    query = """
    SELECT
      *
    FROM
      `{table_name}`
    LIMIT 100
    """.format(
        table_name=table_name
    )

    job_config = bigquery.QueryJobConfig()
    query_job = client.query(query=query, job_config=job_config)
    df = query_job.result().to_dataframe()
    df.to_csv(dataset.path, index=False)

    
    
@component(
    packages_to_install=[
        "xgboost==1.6.2",
        "pandas==1.3.5",
        "joblib==1.1.0",
        "scikit-learn==1.0.2",
    ],
)
def xgboost_training(
    dataset: Input[Dataset],
    model: Output[Model],
    metrics: Output[Metrics],
):
    """Trains an XGBoost classifier.

    Args:
        dataset: The training dataset.

    Returns:
        model: The model artifact stores the model.joblib file.
        metrics: The metrics of the trained model.
    """
    import os

    import joblib
    import pandas as pd
    import xgboost as xgb
    from sklearn.metrics import (accuracy_score, precision_recall_curve,
                                 roc_auc_score)
    from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                         train_test_split)
    from sklearn.preprocessing import LabelEncoder

    # Load the training census dataset
    with open(dataset.path, "r") as train_data:
        raw_data = pd.read_csv(train_data)

    CATEGORICAL_COLUMNS = (
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country",
    )
    LABEL_COLUMN = "income_bracket"
    POSITIVE_VALUE = " >50K"

    # Convert data in categorical columns to numerical values
    encoders = {col: LabelEncoder() for col in CATEGORICAL_COLUMNS}
    for col in CATEGORICAL_COLUMNS:
        raw_data[col] = encoders[col].fit_transform(raw_data[col])

    X = raw_data.drop([LABEL_COLUMN], axis=1).values
    y = raw_data[LABEL_COLUMN] == POSITIVE_VALUE

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    _ = xgb.DMatrix(X_train, label=y_train)
    _ = xgb.DMatrix(X_test, label=y_test)

    params = {
        "reg_lambda": [0, 1],
        "gamma": [1],
        "max_depth": [10],
        "learning_rate": [0.1],
    }

    xgb_model = xgb.XGBClassifier(
        n_estimators=50,
        objective="binary:hinge",
        silent=True,
        nthread=1,
        eval_metric="auc",
    )

    folds = 3
    param_comb = 10

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

    random_search = RandomizedSearchCV(
        xgb_model,
        param_distributions=params,
        n_iter=param_comb,
        scoring="precision",
        n_jobs=4,
        cv=skf.split(X_train, y_train),
        verbose=4,
        random_state=42,
    )

    random_search.fit(X_train, y_train)
    xgb_model_best = random_search.best_estimator_
    predictions = xgb_model_best.predict(X_test)
    score = accuracy_score(y_test, predictions)
    auc = roc_auc_score(y_test, predictions)
    _ = precision_recall_curve(y_test, predictions)

    metrics.log_metric("accuracy", (score * 100.0))
    metrics.log_metric("framework", "xgboost")
    metrics.log_metric("dataset_size", len(raw_data))
    metrics.log_metric("AUC", auc)

    # Export the model to a file
    os.makedirs(model.path, exist_ok=True)
    joblib.dump(xgb_model_best, os.path.join(model.path, "model.joblib"))
    

@component(
    packages_to_install=["google-cloud-aiplatform==1.25.0"],
)
def deploy_xgboost_model(
    model: Input[Model],
    project_id: str,
    vertex_endpoint: Output[Artifact],
    vertex_model: Output[Model],
):
    """Deploys an XGBoost model to Vertex AI Endpoint.

    Args:
        model: The model to deploy.
        project_id: The project ID of the Vertex AI Endpoint.

    Returns:
        vertex_endpoint: The deployed Vertex AI Endpoint.
        vertex_model: The deployed Vertex AI Model.
    """
    from google.cloud import aiplatform

    aiplatform.init(project=project_id)

    deployed_model = aiplatform.Model.upload(
        display_name="census-demo-model",
        artifact_uri=model.uri,
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-6:latest",
    )
    endpoint = deployed_model.deploy(machine_type="n1-standard-4")

    vertex_endpoint.uri = endpoint.resource_name
    vertex_model.uri = deployed_model.resource_name
    
@dsl.pipeline(
    name="census-demo-pipeline",
)
def pipeline():
    """A demo pipeline."""

    create_input_view_task = create_census_view(
        project_id=PROJECT_ID,
        dataset_id=DATASET_ID,
        view_name=VIEW_NAME,
    )

    export_dataset_task = (
        export_dataset(
            project_id=PROJECT_ID,
            dataset_id=DATASET_ID,
            view_name=VIEW_NAME,
        )
        .after(create_input_view_task)
        .set_caching_options(False)
    )

    training_task = xgboost_training(
        dataset=export_dataset_task.outputs["dataset"],
    )

    _ = deploy_xgboost_model(
        project_id=PROJECT_ID,
        model=training_task.outputs["model"],
    )
    
if __name__ == '__main__':
    
    compiler.Compiler().compile(
        pipeline_func=pipeline, package_path="tab_classif_pipeline.json"
    )
    print('Pipeline compilado exitosamente')