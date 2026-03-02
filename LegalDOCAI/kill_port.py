import os
import subprocess
import sys

def kill_port(port):
    print(f"Checking for processes on port {port}...")
    try:
        # Get the PID of the process using the port
        output = subprocess.check_output(f"netstat -ano | findstr :{port}", shell=True).decode()
        lines = output.strip().split('\n')
        pids = set()
        for line in lines:
            if "LISTENING" in line:
                pid = line.strip().split()[-1]
                pids.add(pid)
        
        if not pids:
            print(f"No active process found on port {port}.")
            return

        for pid in pids:
            print(f"Terminating process {pid} using port {port}...")
            subprocess.run(f"taskkill /F /PID {pid}", shell=True)
            print(f"Process {pid} terminated successfully.")
            
    except subprocess.CalledProcessError:
        print(f"No process found on port {port} or error occurred.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    port_to_kill = 8000
    if len(sys.argv) > 1:
        port_to_kill = sys.argv[1]
    kill_port(port_to_kill)
