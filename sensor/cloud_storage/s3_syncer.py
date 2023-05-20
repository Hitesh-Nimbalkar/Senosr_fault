from sensor.constant.s3_bucket import TRAINING_BUCKET_NAME, PREDICTION_BUCKET_NAME, AWS_BUCKET_URL
import os
class S3Sync:


    def sync_folder_to_s3(self,folder,AWS_BUCKET_URL):
        command = f"aws s3 sync {folder} {AWS_BUCKET_URL} "
        os.system(command)

    def sync_folder_from_s3(self,folder,AWS_BUCKET_URL):
        command = f"aws s3 sync  {AWS_BUCKET_URL} {folder} "
        os.system(command)



