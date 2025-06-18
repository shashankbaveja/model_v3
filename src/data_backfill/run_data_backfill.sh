#!/bin/bash

# Data Backfill Script
# This script activates the conda environment and runs data backfill

# Set working directory
cd "/Users/shashankbaveja/Main/Projects/KiteConnectAPI/Model_v3/src/data_backfill"

# Create logs directory if it doesn't exist
mkdir -p logs

# Log file with date
BACKFILL_LOG_FILE="logs/data_backfill_$(date +%Y%m%d).log"
DAILY_BACKFILL_LOG_FILE="logs/data_backfill_daily_$(date +%Y%m%d).log"
CRON_LOG_FILE="logs/cron_backfill_$(date +%Y%m%d).log"

echo "$(date): Starting Minute Data Backfill..." >> "$CRON_LOG_FILE"

# Check if data backfill is already running
if pgrep -f "python data_backfill.py" > /dev/null; then
    echo "$(date): Data Backfill is already running. Exiting." >> "$CRON_LOG_FILE"
    exit 1
fi

# Activate conda environment and run the script
source /opt/anaconda3/bin/activate
conda activate /opt/anaconda3/envs/KiteConnect

# Run data backfill script
echo "$(date): Executing data backfill for last 3 days..." >> "$CRON_LOG_FILE"

python data_backfill.py >> "$BACKFILL_LOG_FILE" 2>&1


echo "$(date): Starting Daily Data Backfill..." >> "$CRON_LOG_FILE"

# Check if data backfill is already running
if pgrep -f "python data_backfill_daily.py" > /dev/null; then
    echo "$(date): Data Backfill is already running. Exiting." >> "$CRON_LOG_FILE"
    exit 1
fi

python data_backfill_daily.py >> "$DAILY_BACKFILL_LOG_FILE" 2>&1

# Check exit status
EXIT_STATUS=$?
if [ $EXIT_STATUS -eq 0 ]; then
    echo "$(date): Data backfill completed successfully." >> "$CRON_LOG_FILE"
else
    echo "$(date): Data backfill failed with exit status: $EXIT_STATUS" >> "$CRON_LOG_FILE"
fi

echo "$(date): Data backfill process complete." >> "$CRON_LOG_FILE" 