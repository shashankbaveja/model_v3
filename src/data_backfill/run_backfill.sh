#!/bin/bash

# Set the working directory
cd /Users/shashankbaveja/Main/Projects/KiteConnectAPI/backfill_cron/

# Create logs directory if it doesn't exist
mkdir -p logs

# Generate log filename with date
LOG_DATE=$(date +%Y%m%d)
LOG_FILE="logs/backfill_${LOG_DATE}.log"

# Function to log with timestamp
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to log script output with timestamp prefix
log_script_output() {
    local script_name=$1
    local start_time=$(date '+%Y-%m-%d %H:%M:%S')
    
    log_message "========================================="
    log_message "Starting $script_name"
    log_message "========================================="
    
    # Run the script and capture both stdout and stderr with timestamps
    python "$script_name" 2>&1 | while IFS= read -r line; do
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] $line" | tee -a "$LOG_FILE"
    done
    
    # Capture exit status
    local exit_status=${PIPESTATUS[0]}
    local end_time=$(date '+%Y-%m-%d %H:%M:%S')
    
    if [ $exit_status -eq 0 ]; then
        log_message "$script_name completed successfully"
        log_message "Duration: $start_time to $end_time"
        log_message "========================================="
    else
        log_message "ERROR: $script_name failed with exit code $exit_status"
        log_message "Duration: $start_time to $end_time"
        log_message "========================================="
    fi
    
    return $exit_status
}

# Start logging
log_message "Starting backfill cron job"
log_message "Working directory: $(pwd)"

# Activate conda environment
log_message "Activating conda environment..."
source /opt/anaconda3/bin/activate
conda activate KiteConnect

if [ $? -eq 0 ]; then
    log_message "Conda environment 'KiteConnect' activated successfully"
else
    log_message "ERROR: Failed to activate conda environment"
    exit 1
fi

# Run the first Python script
log_script_output "data_backfill_daily.py"

# Check if first script completed successfully
if [ $? -eq 0 ]; then
    # Run the second Python script
    log_script_output "data_backfill.py"
    
    if [ $? -eq 0 ]; then
        log_message "🎉 All backfill jobs completed successfully!"
        log_message "Job finished at: $(date)"
        exit 0
    else
        log_message "❌ Job failed at data_backfill.py"
        exit 1
    fi
else
    log_message "❌ Job failed at data_backfill_daily.py - skipping data_backfill.py"
    exit 1
fi