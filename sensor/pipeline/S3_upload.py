


from sensor.constant.s3_bucket import TRAINING_BUCKET_NAME
from sensor.constant.training_pipeline import SAVED_MODEL_DIR
from sensor.exception import ApplicationException
import sys
from sensor.cloud_storage.s3_syncer import S3Sync
from sensor.entity.config_entity import TrainingPipelineConfig


class S3_upload:
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()
        self.s3_sync = S3Sync()
        
    def sync_artifact_dir_to_s3(self):
        try:
            aws_buket_url = f"s3://{TRAINING_BUCKET_NAME}/artifact/{self.training_pipeline_config.timestamp}"
            self.s3_sync.sync_folder_to_s3(folder = self.training_pipeline_config.artifact_dir,aws_buket_url=aws_buket_url)
        except Exception as e:
            raise ApplicationException(e,sys)
            
    def sync_saved_model_dir_to_s3(self):
        try:
            aws_buket_url = f"s3://{TRAINING_BUCKET_NAME}/{SAVED_MODEL_DIR}"
            self.s3_sync.sync_folder_to_s3(folder = SAVED_MODEL_DIR,aws_buket_url=aws_buket_url)
        except Exception as e:
            raise ApplicationException(e,sys)
        
    def upload(self):
        
        try:
            self.sync_artifact_dir_to_s3()
            self.sync_saved_model_dir_to_s3
        except Exception as e:
            raise ApplicationException(e,sys)