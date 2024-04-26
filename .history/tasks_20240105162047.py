import argparse
import logging
import threading

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def print_numbers(start, end, task_name):
    logging.info(f"{task_name}: Printing numbers from {start} to {end}")
    for number in range(start, end + 1):
        logging.info(f"{task_name}: {number}")
    logging.info(f"{task_name}: Task completed.")

def print_every_other_number(start, end, task_name):
    logging.info(f"{task_name}: Printing every other number from {start} to {end}")
    for number in range(start, end + 1, 2):
        logging.info(f"{task_name}: {number}")
    logging.info(f"{task_name}: Task completed.")

def run_task(task_function, start, end, task_name):
    task_thread = threading.Thread(target=task_function, args=(start, end, task_name))
    task_thread.start()
    task_thread.join()

def main():
    parser = argparse.ArgumentParser(description='Task Runner for Multiple Tasks')
    parser.add_argument('--task', type=string, choices=['train', 'import'], required=True, help='Task number to run (1 or 2)')
    parser.add_argument('--start', type=int, default=1, help='Start of the number range')
    parser.add_argument('--end', type=int, default=10, help='End of the number range')

    args = parser.parse_args()

    task_name = f"Task {args.task}"
    if args.task == "train":
        run_task(print_numbers, args.start, args.end, task_name)
    elif args.task == "import":
        run_task(print_every_other_number, args.start, args.end, task_name)

if __name__ == "__main__":
    main()