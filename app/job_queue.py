import logging
import os
import time

from redis import Redis
from redis.exceptions import ConnectionError

from rq import Queue
from rq.job import Job

# Jobs: inspect
from features.inpect import inspect

# Get Redis configuration from environment variables or use defaults
REDIS_HOST = os.getenv('REDIS_HOST', 'redis')  # Use 'redis' as default hostname in Docker
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_DB = int(os.getenv('REDIS_DB', 0))

def get_redis_connection():
    """Create Redis connection with retry logic"""
    retry_count = 0
    max_retries = 5
    
    while retry_count < max_retries:
        try:
            redis_conn = Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                socket_connect_timeout=5
            )
            redis_conn.ping()  # Test the connection
            logging.info(f"Redis connection established {REDIS_HOST}:{REDIS_PORT}/{REDIS_DB} ")
            return redis_conn
        except ConnectionError as e:
            retry_count += 1
            if retry_count == max_retries:
                logging.error(f"Could not connect to Redis at {REDIS_HOST}:{REDIS_PORT}/{REDIS_DB} after {max_retries} attempts")
                raise
            logging.warning(f"Redis connection attempt {retry_count} failed. Retrying in 5 seconds...")
            time.sleep(5)

# Initialize Redis connection
redis_conn = get_redis_connection()
detection_inspection_queue = Queue('detection-inspection-queue', connection=redis_conn)

#
# Callers of enqueue_inspection_job should get aync_job_queue from os.getenv('JOB_QUEUE', False)
#
def enqueue_inspection_job( detection, frame = None, video_path = None, aync_job_queue = False ):
    
    job_id = None
    job_status = None

    if aync_job_queue: # Schedule as asynchronous job into queue
        try:

            job = detection_inspection_queue.enqueue(
                inspect,
                detection,
                frame,
                video_path,
                job_timeout='1m',
                result_ttl=500
            )

            logging.debug(f"Enqueued job {job.id} with detection: {detection} from {video_path}")
        
            # Get job status
            job_status = job.get_status()
            job_id = job.id

            logging.info(f"Job {job.id} status: {job_status}")
        
        except Exception as e:
            job_status = str(e)
            job_id = None

            logging.error(f"Error enqueue_inspection_job : {str(e)}")

    else: # Call synchronously 
        inspect( detection, frame, video_path )

    return job_id, job_status
