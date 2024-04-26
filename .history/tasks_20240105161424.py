import argparse
import logging
import threading

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def print_numbers(start, end):
    logging.info(f"Task 1: Printing numbers from {start} to {end}")
    for number in range(start, end + 1):
        logging.info(number)
    logging.info("Task 1 completed.")

def print_every_other_number(start, end):
    logging.info(f"Task 2: Printing every other number from {start} to {end}")
    for number in range(start, end + 1, 2):
        logging.info(number)
    logging.info("Task 2 completed.")

def run_task(task_function, start, end):
    task_thread = threading.Thread(target=task_function, args=(start, end))
    task_thread.start()
    task_thread.join()

def main():
    parser = argparse.ArgumentParser(description='Task Runner for Multiple Tasks')
    parser.add_argument('--task', type=int, choices=[1, 2], required=True, help='Task number to run (1 or 2)')
    parser.add_argument('--start', type=int, default=1, help='Start of the number range')
    parser.add_argument('--end', type=int, default=10, help='End of the number range')

    args = parser.parse_args()

    if args.task == 1:
        run_task(print_numbers, args.start, args.end)
    elif args.task == 2:
        run_task(print_every_other_number, args.start, args.end)

if __name__ == "__main__":
    main()