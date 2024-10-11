import subprocess
import re

# Specify the Python file to run
script_to_run = "test1.py"  # Replace with the path to your Python script

# Number of times to run the script
num_runs = 25

# To store run times and mismatches for averaging
run_times_list = []
mismatches_list = []

for i in range(num_runs):
    try:
        # Run the Python script and capture the output
        result = subprocess.run(
            ["python", script_to_run],
            capture_output=True,  # Capture standard output and error
            text=True,           # Return output as string
            check=True           # Raise an error if the command fails
        )
        # Split the output into lines and get the last line
        last_line = result.stdout.strip().splitlines()[-1]

        # Extract run time and mismatches using regular expressions
        time_match = re.search(r"Run Time:\s*([\d.]+)\s*seconds", last_line)
        mismatch_match = re.search(r"Mismatches:\s*(\d+)", last_line)

        if time_match and mismatch_match:
            run_time = float(time_match.group(1))
            mismatches = int(mismatch_match.group(1))
            run_times_list.append(run_time)
            mismatches_list.append(mismatches)

        # Print to console for confirmation
        print(f"Run {i + 1}: {last_line}")

    except subprocess.CalledProcessError as e:
        print(f"Run {i + 1} failed: {e}")

# Calculate and print the average run time and mismatches
if run_times_list and mismatches_list:
    avg_run_time = sum(run_times_list) / len(run_times_list)
    avg_mismatches = sum(mismatches_list) / len(mismatches_list)

    print(f"\nAverage Run Time: {avg_run_time:.2f} seconds")
    print(f"Average Mismatches: {avg_mismatches:.2f}")

print(f"\nCompleted {num_runs} runs.")
