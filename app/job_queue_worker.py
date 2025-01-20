import logging
import os
import time

from redis import Redis
from rq import Worker, Queue
from rq.exceptions import NoSuchJobError
from rq.registry import StartedJobRegistry

import signal
import sys

# Get log level from the environment variable
log_level = os.getenv("LOG_LEVEL", "INFO").upper()

# Configure logging
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Configure logging for RQ
rq_logger = logging.getLogger("rq.worker")
rq_logger.setLevel(getattr(logging, log_level, logging.WARNING))

# Environment variables for Redis configuration
REDIS_HOST = os.getenv("REDIS_HOST", "redis")  # Default to 'redis' for Docker
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
QUEUE_NAME = os.getenv("QUEUE_NAME", "detection-inspection-queue")  # Queue to process

def get_redis_connection():
    """Establish a connection to the Redis server."""
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
            redis_conn.ping()
            logging.info(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}")
            return redis_conn
        except ConnectionError:
            retry_count += 1
            logging.warning(
                f"Failed to connect to Redis (attempt {retry_count}/{max_retries}). Retrying in 5 seconds..."
            )
            time.sleep(5)

    raise ConnectionError(f"Could not connect to Redis after {max_retries} attempts.")

def close_redis_connection(redis_conn):
    if redis_conn:
        logging.info("Closing Redis connection...")
        redis_conn.close()


def shutdown_worker(worker):
    if worker:
        logging.info("Shutting down worker...")
        worker.request_stop()
        worker._shutdown()  # Clean up worker internals


def handle_interrupted_jobs(queue):
    logging.info("Checking for interrupted jobs...")
    registry = StartedJobRegistry(queue.name, connection=queue.connection)
    job_ids = registry.get_job_ids()

    for job_id in job_ids:
        job = queue.fetch_job(job_id)
        if job:
            logging.warning(f"Marking interrupted job {job.id} as failed...")
            job.set_status('failed')
            queue.failed_job_registry.add(job, ttl=queue.failed_job_registry.DEFAULT_TTL)


def release_resources():
    logging.info("Releasing external resources...")
    # Example: Closing database connections, files, etc.


def handle_shutdown(worker, redis_conn, queue):
    logging.info("Shutting down gracefully...")
    shutdown_worker(worker)
    handle_interrupted_jobs(queue)
    close_redis_connection(redis_conn)
    release_resources()
    logging.info("Cleanup complete. Exiting.")
    sys.exit(0)


def main():

    logging.info("Worker started with log level: %s", log_level)

    '''
    Start the RQ worker to process jobs.
    '''
    redis_conn = get_redis_connection()
    queue = Queue(QUEUE_NAME, connection=redis_conn)
    worker = Worker([queue], connection=redis_conn)

    signal.signal(signal.SIGINT , lambda s, f: handle_shutdown(worker, redis_conn, queue))
    signal.signal(signal.SIGTERM, lambda s, f: handle_shutdown(worker, redis_conn, queue))

    logging.info(f"Starting worker for queue: {QUEUE_NAME}")
   
    try:
        worker.work(with_scheduler=True)  # Enable the RQ scheduler
    except NoSuchJobError as e:
        logging.error(f"Job error: {e}")
    except KeyboardInterrupt:
        logging.info("Worker stopped manually.")
    except Exception as e:
        logging.error(f"Unexpected error: {e}", exc_info=True)

if __name__ == "__main__":
    main()
