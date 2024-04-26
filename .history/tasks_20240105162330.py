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

def run_task(task_function, start, end, task_name):
    task_thread = threading.Thread(target=task_function, args=(start, end, task_name))
    task_thread.start()
    task_thread.join()

def main():
    parser = argparse.ArgumentParser(description='Task Runner for Named Tasks')
    parser.add_argument('--task', type=str, choices=['train', 'import'], required=True, help='Name of the task to run (train or import)')
    parser.add_argument('--start', type=int, default=1, help='Start of the range')
    parser.add_argument('--end', type=int, default=10, help='End of the range')

    args = parser.parse_args()

    task_name = args.task.capitalize()
    if args.task == 'train':
        run_task(train_task, args.start, args.end, task_name)
    elif args.task == 'import':
        run_task(import_task, args.start, args.end, task_name)

if __name__ == "__main__":
    main()