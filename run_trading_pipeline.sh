#!/bin/bash

# Trading Pipeline Automation Script
# Runs the complete trading pipeline every weekday at 3:00 PM

# Set script configuration
SCRIPT_DIR="/Users/shashankbaveja/Main/Projects/KiteConnectAPI/Model_v3"
CONDA_ENV="/opt/anaconda3/envs/KiteConnect"
PYTHON_SCRIPT="src/run_live.py"
LOG_DIR="$SCRIPT_DIR/logs/automation"

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Generate timestamp for this run
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
AUTOMATION_LOG="$LOG_DIR/automation_$TIMESTAMP.log"

# Function to log messages
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$AUTOMATION_LOG"
}

# Start logging
log_message "======================================"
log_message "Starting Automated Trading Pipeline"
log_message "======================================"

# Change to project directory
log_message "Changing to project directory: $SCRIPT_DIR"
cd "$SCRIPT_DIR" || {
    log_message "ERROR: Failed to change to project directory"
    exit 1
}

# Verify Python environment exists
if [ ! -f "$CONDA_ENV/bin/python" ]; then
    log_message "ERROR: Python environment not found at $CONDA_ENV/bin/python"
    exit 1
fi

# Verify Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    log_message "ERROR: Python script not found at $PYTHON_SCRIPT"
    exit 1
fi

# Check if it's a weekday (Monday=1, Friday=5)
DAY_OF_WEEK=$(date +%u)
if [ "$DAY_OF_WEEK" -gt 5 ]; then
    log_message "INFO: Today is weekend (day $DAY_OF_WEEK), skipping pipeline execution"
    exit 0
fi

log_message "INFO: Today is weekday (day $DAY_OF_WEEK), proceeding with pipeline"

# Set environment variables if needed
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Run the trading pipeline
log_message "Executing trading pipeline..."
log_message "Command: $CONDA_ENV/bin/python $PYTHON_SCRIPT"
log_message "---- PIPELINE OUTPUT START ----"

# Capture all output (stdout and stderr) and append to automation log
if "$CONDA_ENV/bin/python" "$PYTHON_SCRIPT" 2>&1 | tee -a "$AUTOMATION_LOG"; then
    log_message "---- PIPELINE OUTPUT END ----"
    log_message "SUCCESS: Trading pipeline completed successfully"
    
    # Optional: Send success notification (uncomment if needed)
    # echo "Trading pipeline completed successfully at $(date)" | mail -s "Trading Pipeline Success" your-email@example.com
    
    exit_code=0
else
    log_message "---- PIPELINE OUTPUT END ----"
    log_message "ERROR: Trading pipeline failed with exit code $?"
    
    # Optional: Send failure notification (uncomment if needed)
    # echo "Trading pipeline failed at $(date). Check logs at $AUTOMATION_LOG" | mail -s "Trading Pipeline FAILED" your-email@example.com
    
    exit_code=1
fi

log_message "======================================"
log_message "Automated Trading Pipeline Finished"
log_message "Exit Code: $exit_code"
log_message "Log saved to: $AUTOMATION_LOG"
log_message "======================================"

exit $exit_code 