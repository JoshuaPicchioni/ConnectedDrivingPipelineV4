#!/usr/bin/env python3
"""
Pipeline Queue (pq) - Robust pipeline job queue system
Usage:
  pq status          - Show queue status and current job progress
  pq add <pipeline>  - Add pipeline to queue
  pq list            - List all pipelines available
  pq logs [job_id]   - Tail logs for current/specific job
  pq cancel <job_id> - Cancel a queued job
  pq history         - Show completed jobs
  pq start           - Start the queue daemon
  pq stop            - Stop the queue daemon
"""

import json
import os
import sys
import subprocess
import signal
import time
from datetime import datetime
from pathlib import Path

# Paths
REPO_DIR = Path.home() / "repos" / "ConnectedDrivingPipelineV4"
QUEUE_DIR = Path.home() / ".pipeline-queue"
QUEUE_FILE = QUEUE_DIR / "queue.json"
STATE_FILE = QUEUE_DIR / "state.json"
PID_FILE = QUEUE_DIR / "daemon.pid"
RESULTS_BASE = Path("/var/www/static/pipeline-results")
DASK_CONFIG = REPO_DIR / "configs" / "dask" / "64gb-production.yml"

# Colors
class C:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

def init():
    """Initialize queue directories and files."""
    QUEUE_DIR.mkdir(exist_ok=True)
    if not QUEUE_FILE.exists():
        save_queue([])
    if not STATE_FILE.exists():
        save_state({"current_job": None, "history": []})

def load_queue():
    try:
        return json.loads(QUEUE_FILE.read_text())
    except:
        return []

def save_queue(queue):
    QUEUE_FILE.write_text(json.dumps(queue, indent=2))

def load_state():
    try:
        return json.loads(STATE_FILE.read_text())
    except:
        return {"current_job": None, "history": []}

def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2))

def generate_job_id():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def find_pipeline(name):
    """Find pipeline file by partial name match."""
    pipelines = list(REPO_DIR.glob("MClassifier*.py"))
    matches = [p for p in pipelines if name in p.name]
    if len(matches) == 1:
        return matches[0].name
    elif len(matches) > 1:
        print(f"{C.YELLOW}Multiple matches:{C.END}")
        for m in matches[:10]:
            print(f"  {m.name}")
        return None
    return None

def get_progress_from_log(log_path):
    """Parse log file for progress info."""
    if not log_path.exists():
        return None
    
    try:
        # Get last 100 lines
        result = subprocess.run(
            ["tail", "-100", str(log_path)],
            capture_output=True, text=True
        )
        lines = result.stdout.strip().split('\n')
        
        progress = {
            "stage": "Unknown",
            "last_activity": None,
            "rows_processed": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        for line in reversed(lines):
            if "DataGatherer:" in line and "line" in line:
                # Extract row count
                try:
                    parts = line.split("line ")
                    if len(parts) > 1:
                        progress["rows_processed"] = int(parts[1].strip())
                        progress["stage"] = "Data Gathering"
                except:
                    pass
            elif "ConnectedDrivingLargeDataCleaner:" in line:
                progress["stage"] = "Data Cleaning"
            elif "Training" in line or "Fitting" in line:
                progress["stage"] = "Model Training"
            elif "Evaluating" in line or "Testing" in line:
                progress["stage"] = "Evaluation"
            elif "cache_hits=" in line.lower() or "hit_rate=" in line:
                progress["stage"] = "Data Cleaning (cached)"
            
            # Get timestamp
            if progress["last_activity"] is None and "LogLevel" in line:
                try:
                    ts = line.split("LogLevel.INFO ")[1].split(" ")[0:2]
                    progress["last_activity"] = " ".join(ts)
                except:
                    pass
        
        return progress
    except:
        return None

def format_duration(seconds):
    """Format seconds into human readable duration."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds // 60:.0f}m {seconds % 60:.0f}s"
    else:
        hours = seconds // 3600
        mins = (seconds % 3600) // 60
        return f"{hours:.0f}h {mins:.0f}m"

def cmd_status():
    """Show queue status and current job progress."""
    init()
    state = load_state()
    queue = load_queue()
    
    print(f"\n{C.BOLD}═══════════════════════════════════════════════════════════════{C.END}")
    print(f"{C.BOLD}  📊 Pipeline Queue Status{C.END}")
    print(f"{C.BOLD}═══════════════════════════════════════════════════════════════{C.END}\n")
    
    # Check if daemon is running
    daemon_running = False
    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text().strip())
            os.kill(pid, 0)  # Check if process exists
            daemon_running = True
        except:
            pass
    
    print(f"  Daemon: {C.GREEN}Running ✓{C.END}" if daemon_running else f"  Daemon: {C.RED}Stopped ✗{C.END}")
    print(f"  Queued: {C.CYAN}{len(queue)}{C.END} jobs")
    
    # Current job
    current = state.get("current_job")
    if current:
        print(f"\n{C.BOLD}▶ CURRENT JOB{C.END}")
        print(f"  {C.CYAN}ID:{C.END} {current['id']}")
        print(f"  {C.CYAN}Pipeline:{C.END} {current['pipeline'][:60]}...")
        print(f"  {C.CYAN}Started:{C.END} {current.get('started_at', 'Unknown')}")
        
        # Calculate runtime
        if current.get('started_at'):
            try:
                start = datetime.strptime(current['started_at'], "%Y-%m-%d %H:%M:%S")
                runtime = (datetime.now() - start).total_seconds()
                print(f"  {C.CYAN}Runtime:{C.END} {format_duration(runtime)}")
            except:
                pass
        
        # Get progress from log
        log_path = Path(current.get('results_dir', '')) / "pipeline.log"
        progress = get_progress_from_log(log_path)
        if progress:
            print(f"  {C.CYAN}Stage:{C.END} {C.YELLOW}{progress['stage']}{C.END}")
            if progress['rows_processed']:
                print(f"  {C.CYAN}Rows:{C.END} {progress['rows_processed']:,}")
            if progress['last_activity']:
                print(f"  {C.CYAN}Last Activity:{C.END} {progress['last_activity']}")
        
        print(f"  {C.CYAN}Results:{C.END} {current.get('results_dir', 'N/A')}")
    else:
        print(f"\n{C.YELLOW}No job currently running{C.END}")
    
    # Queue
    if queue:
        print(f"\n{C.BOLD}📋 QUEUED JOBS{C.END}")
        for i, job in enumerate(queue, 1):
            short_name = job['pipeline'][:50] + "..." if len(job['pipeline']) > 50 else job['pipeline']
            print(f"  {i}. [{job['id']}] {short_name}")
    
    # Recent history
    history = state.get("history", [])[-5:]
    if history:
        print(f"\n{C.BOLD}📜 RECENT HISTORY{C.END}")
        for job in reversed(history):
            status_icon = "✓" if job.get('exit_code') == 0 else "✗"
            status_color = C.GREEN if job.get('exit_code') == 0 else C.RED
            duration = job.get('duration', 'N/A')
            print(f"  {status_color}{status_icon}{C.END} [{job['id']}] {job['pipeline'][:40]}... ({duration})")
    
    print(f"\n{C.BOLD}═══════════════════════════════════════════════════════════════{C.END}\n")

def cmd_add(pipeline_name):
    """Add a pipeline to the queue."""
    init()
    
    # Check if it's a full filename or partial
    if pipeline_name.endswith('.py') and (REPO_DIR / pipeline_name).exists():
        pipeline = pipeline_name
    else:
        pipeline = find_pipeline(pipeline_name)
        if not pipeline:
            print(f"{C.RED}Pipeline not found: {pipeline_name}{C.END}")
            return
    
    queue = load_queue()
    job = {
        "id": generate_job_id(),
        "pipeline": pipeline,
        "added_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "queued"
    }
    queue.append(job)
    save_queue(queue)
    
    print(f"{C.GREEN}✓ Added to queue:{C.END} {pipeline}")
    print(f"  Job ID: {job['id']}")
    print(f"  Position: {len(queue)}")

def cmd_list():
    """List available pipelines."""
    pipelines = sorted(REPO_DIR.glob("MClassifier*.py"))
    
    print(f"\n{C.BOLD}Available Pipelines ({len(pipelines)} total){C.END}\n")
    
    # Group by attack type
    groups = {
        "RandOffset": [],
        "ConstPos": [],
        "RandomPos": [],
        "RandOverride": [],
        "ConstOffset": [],
        "PositionSwap": [],
        "Other": []
    }
    
    for p in pipelines:
        name = p.name
        categorized = False
        for key in groups:
            if key in name:
                groups[key].append(name)
                categorized = True
                break
        if not categorized:
            groups["Other"].append(name)
    
    for group, items in groups.items():
        if items:
            print(f"{C.CYAN}{group}{C.END} ({len(items)})")
            for item in items[:5]:
                print(f"  {item[:70]}...")
            if len(items) > 5:
                print(f"  ... and {len(items) - 5} more")
            print()

def cmd_logs(job_id=None):
    """Tail logs for current or specific job."""
    init()
    state = load_state()
    
    if job_id:
        # Find job in history
        for job in state.get("history", []):
            if job["id"] == job_id:
                log_path = Path(job["results_dir"]) / "pipeline.log"
                break
        else:
            print(f"{C.RED}Job not found: {job_id}{C.END}")
            return
    else:
        current = state.get("current_job")
        if not current:
            print(f"{C.YELLOW}No job currently running{C.END}")
            return
        log_path = Path(current["results_dir"]) / "pipeline.log"
    
    if not log_path.exists():
        print(f"{C.RED}Log file not found: {log_path}{C.END}")
        return
    
    print(f"{C.CYAN}Tailing: {log_path}{C.END}\n")
    os.execvp("tail", ["tail", "-f", str(log_path)])

def cmd_cancel(job_id):
    """Cancel a queued job."""
    init()
    queue = load_queue()
    
    new_queue = [j for j in queue if j["id"] != job_id]
    if len(new_queue) == len(queue):
        print(f"{C.RED}Job not found in queue: {job_id}{C.END}")
        return
    
    save_queue(new_queue)
    print(f"{C.GREEN}✓ Cancelled job: {job_id}{C.END}")

def cmd_history():
    """Show completed jobs history."""
    init()
    state = load_state()
    history = state.get("history", [])
    
    if not history:
        print(f"{C.YELLOW}No job history{C.END}")
        return
    
    print(f"\n{C.BOLD}Job History ({len(history)} jobs){C.END}\n")
    
    for job in reversed(history[-20:]):
        status_icon = "✓" if job.get('exit_code') == 0 else "✗"
        status_color = C.GREEN if job.get('exit_code') == 0 else C.RED
        print(f"{status_color}{status_icon}{C.END} [{job['id']}]")
        print(f"  Pipeline: {job['pipeline'][:60]}...")
        print(f"  Duration: {job.get('duration', 'N/A')}")
        print(f"  Results: {job.get('results_dir', 'N/A')}")
        print()

def cmd_start():
    """Start the queue daemon."""
    init()
    
    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text().strip())
            os.kill(pid, 0)
            print(f"{C.YELLOW}Daemon already running (PID {pid}){C.END}")
            return
        except:
            pass
    
    # Start daemon
    daemon_script = QUEUE_DIR / "daemon.py"
    
    # Write daemon script
    daemon_code = '''#!/usr/bin/env python3
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_DIR = Path.home() / "repos" / "ConnectedDrivingPipelineV4"
QUEUE_DIR = Path.home() / ".pipeline-queue"
QUEUE_FILE = QUEUE_DIR / "queue.json"
STATE_FILE = QUEUE_DIR / "state.json"
PID_FILE = QUEUE_DIR / "daemon.pid"
RESULTS_BASE = Path("/var/www/static/pipeline-results")
DASK_CONFIG = REPO_DIR / "configs" / "dask" / "64gb-production.yml"

def load_queue():
    try:
        return json.loads(QUEUE_FILE.read_text())
    except:
        return []

def save_queue(queue):
    QUEUE_FILE.write_text(json.dumps(queue, indent=2))

def load_state():
    try:
        return json.loads(STATE_FILE.read_text())
    except:
        return {"current_job": None, "history": []}

def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2))

def format_duration(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds // 60:.0f}m {seconds % 60:.0f}s"
    else:
        hours = seconds // 3600
        mins = (seconds % 3600) // 60
        return f"{hours:.0f}h {mins:.0f}m"

def run_job(job):
    """Run a single pipeline job."""
    state = load_state()
    
    # Create results directory
    results_dir = RESULTS_BASE / job["id"]
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Update state
    job["status"] = "running"
    job["started_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    job["results_dir"] = str(results_dir)
    state["current_job"] = job
    save_state(state)
    
    # Write job info
    (results_dir / "job.json").write_text(json.dumps(job, indent=2))
    
    # Prepare environment
    env = os.environ.copy()
    env["DASK_CONFIG"] = str(DASK_CONFIG)
    
    # Build command
    log_path = results_dir / "pipeline.log"
    
    cmd = f"""
    cd {REPO_DIR}
    source venv/bin/activate
    export DASK_CONFIG="{DASK_CONFIG}"
    python {job["pipeline"]} 2>&1 | tee {log_path}
    """
    
    # Run pipeline
    start_time = time.time()
    result = subprocess.run(
        ["bash", "-c", cmd],
        cwd=REPO_DIR,
        env=env
    )
    duration = time.time() - start_time
    
    # Copy results
    subprocess.run([
        "bash", "-c",
        f"""
        cd {REPO_DIR}
        find data -name "*.csv" -size -100M -exec cp {{}} {results_dir}/ \\; 2>/dev/null
        find data -name "*.json" -exec cp {{}} {results_dir}/ \\; 2>/dev/null
        find data -name "*.png" -exec cp {{}} {results_dir}/ \\; 2>/dev/null
        find . -maxdepth 1 -name "*.csv" -size -100M -exec cp {{}} {results_dir}/ \\; 2>/dev/null
        find . -maxdepth 1 -name "*.png" -exec cp {{}} {results_dir}/ \\; 2>/dev/null
        """
    ])
    
    # Update state
    state = load_state()
    job["status"] = "completed" if result.returncode == 0 else "failed"
    job["finished_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    job["exit_code"] = result.returncode
    job["duration"] = format_duration(duration)
    
    state["current_job"] = None
    state["history"].append(job)
    save_state(state)
    
    # Write completion info
    (results_dir / "README.txt").write_text(f"""Pipeline: {job["pipeline"]}
Job ID: {job["id"]}
Started: {job["started_at"]}
Finished: {job["finished_at"]}
Duration: {job["duration"]}
Exit Code: {result.returncode}
""")
    
    return result.returncode == 0

def main():
    # Write PID
    PID_FILE.write_text(str(os.getpid()))
    
    print(f"Pipeline Queue Daemon started (PID {os.getpid()})")
    
    try:
        while True:
            state = load_state()
            
            # Skip if job already running
            if state.get("current_job"):
                time.sleep(10)
                continue
            
            # Get next job
            queue = load_queue()
            if not queue:
                time.sleep(5)
                continue
            
            # Pop next job
            job = queue.pop(0)
            save_queue(queue)
            
            print(f"Starting job: {job['id']} - {job['pipeline']}")
            success = run_job(job)
            print(f"Job completed: {job['id']} - {'Success' if success else 'Failed'}")
            
    except KeyboardInterrupt:
        print("Daemon stopping...")
    finally:
        PID_FILE.unlink(missing_ok=True)

if __name__ == "__main__":
    main()
'''
    daemon_script.write_text(daemon_code)
    daemon_script.chmod(0o755)
    
    # Start daemon in background
    log_file = QUEUE_DIR / "daemon.log"
    subprocess.Popen(
        ["nohup", "python3", str(daemon_script)],
        stdout=open(log_file, "a"),
        stderr=subprocess.STDOUT,
        start_new_session=True
    )
    
    time.sleep(1)
    
    if PID_FILE.exists():
        pid = PID_FILE.read_text().strip()
        print(f"{C.GREEN}✓ Daemon started (PID {pid}){C.END}")
        print(f"  Log: {log_file}")
    else:
        print(f"{C.RED}Failed to start daemon{C.END}")

def cmd_stop():
    """Stop the queue daemon."""
    if not PID_FILE.exists():
        print(f"{C.YELLOW}Daemon not running{C.END}")
        return
    
    try:
        pid = int(PID_FILE.read_text().strip())
        os.kill(pid, signal.SIGTERM)
        print(f"{C.GREEN}✓ Daemon stopped (PID {pid}){C.END}")
        PID_FILE.unlink(missing_ok=True)
    except ProcessLookupError:
        print(f"{C.YELLOW}Daemon was not running{C.END}")
        PID_FILE.unlink(missing_ok=True)
    except Exception as e:
        print(f"{C.RED}Error stopping daemon: {e}{C.END}")

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    
    cmd = sys.argv[1]
    
    if cmd == "status":
        cmd_status()
    elif cmd == "add" and len(sys.argv) > 2:
        cmd_add(sys.argv[2])
    elif cmd == "list":
        cmd_list()
    elif cmd == "logs":
        job_id = sys.argv[2] if len(sys.argv) > 2 else None
        cmd_logs(job_id)
    elif cmd == "cancel" and len(sys.argv) > 2:
        cmd_cancel(sys.argv[2])
    elif cmd == "history":
        cmd_history()
    elif cmd == "start":
        cmd_start()
    elif cmd == "stop":
        cmd_stop()
    else:
        print(__doc__)

if __name__ == "__main__":
    main()
