import subprocess
import datetime
import os

def run_indexing_with_live_output_and_error_log():
    cmd = ["python3", "log_matcher.py", "--index"]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    stdout_lines = []
    stderr_lines = []

    # Read output line by line and print live
    for line in process.stdout:
        print(line, end="")
        stdout_lines.append(line)

    for line in process.stderr:
        print(line, end="", flush=True)  # Also show errors live
        stderr_lines.append(line)

    process.wait()

    if process.returncode != 0:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        error_dir = "logs"
        os.makedirs(error_dir, exist_ok=True)
        error_path = os.path.join(error_dir, f"indexing_error_{timestamp}.log")

        with open(error_path, "w") as f:
            f.write("COMMAND: " + " ".join(cmd) + "\n")
            f.write("\n--- STDERR ---\n")
            f.writelines(stderr_lines)
            f.write("\n--- STDOUT ---\n")
            f.writelines(stdout_lines)

        print(f"\n❌ Error occurred. Logged to: {error_path}")
    else:
        print("\n✅ Indexing completed successfully.")
