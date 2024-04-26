import argparse
import logging
import threading

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_task(start, end, task_name):
    logging.info(f"{task_name}: Training task started with range {start} to {end}")
    for number in range(start, end + 1):
        logging.info(f"{task_name}: Training on data chunk {number}")
    logging.info(f"{task_name}: Training task completed.")

def import_task(start, end, task_name):
    logging.info(f"{task_name}: Import task started with range {start} to {end}")
    for number in range(start, end + 1, 2):
        logging.info(f"{task_name}: Importing data chunk {number}")
    logging.info(f"{task_name}: Import task completed.")

def run_task(task_function, start, end, task_name, delay):
    logging.info(f"Scheduling {task_name} to start in {delay} seconds")
    timer = threading.Timer(delay, task_function, args=[start, end, task_name])
    timer.start()

def main():
    parser = argparse.ArgumentParser(description='Task Runner with Scheduling Option')
    parser.add_argument('--task', type=str, choices=['train', 'import'], required=True, help='Name of the task to run (train or import)')
    parser.add_argument('--start', type=int, default=1, help='Start of the range')
    parser.add_argument('--end', type=int, default=10, help='End of the range')
    parser.add_argument('--delay', type=int, default=0, help='Delay in seconds before the task starts')

    args = parser.parse_args()

    task_name = args.task.capitalize()
    run_task(train_task if args.task == 'train' else import_task, args.start, args.end, task_name, args.delay)

if __name__ == "__main__":
    main()