from datetime import datetime
from google.cloud import aiplatform


if __name__ == '__main__':
    TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")

    job = aiplatform.PipelineJob(
        display_name="mlops-xgboost",
        template_path="tab_classif_pipeline.json",
        job_id="mlops-xgboost-{0}".format(TIMESTAMP),
        enable_caching=True
    )

    job.submit()
    
    print('Pipeline successfully submitted')