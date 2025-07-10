Yes — you can capture errors from a subprocess and write them to a separate error log file, dynamically generating the file only when an error occurs.


---

✅ Full Example

Here’s how to run a subprocess for indexing and log errors only if they occur:

import subprocess
import datetime
import os

def run_indexing_with_error_log():
    cmd = ["python3", "log_matcher.py", "--index"]
    
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        # Only create error log if there was an error
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        error_filename = f"indexing_error_{timestamp}.log"
        error_dir = "logs"
        os.makedirs(error_dir, exist_ok=True)
        error_path = os.path.join(error_dir, error_filename)

        with open(error_path, "w") as f:
            f.write("COMMAND: " + " ".join(cmd) + "\n")
            f.write("\n--- STDERR ---\n")
            f.write(result.stderr)
            f.write("\n--- STDOUT ---\n")
            f.write(result.stdout)

        print(f"❌ Error occurred. Logged to: {error_path}")
    else:
        print("✅ Indexing completed successfully.")


---

🔍 What this does:

Uses subprocess.run(..., capture_output=True) to capture both stdout and stderr.

If returncode != 0, it:

Creates a folder logs/ if it doesn’t exist.

Writes a timestamped error log file.


No error file is created if indexing runs fine.



---

🧪 Output Example:

If an error happens, you'll get:

❌ Error occurred. Logged to: logs/indexing_error_20250710_174552.log


---

Let me know if you want the logs in JSON format or need this integrated into a monitoring system or dashboard.

