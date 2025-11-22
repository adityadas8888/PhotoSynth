#!/bin/bash

# PhotoSynth Power Schedule Setup (Run on 3090)
# This script sets up cron jobs to wake the machine at 2 AM and suspend it at 8 AM.

# Ensure rtcwake is installed
if ! command -v rtcwake &> /dev/null; then
    echo "Installing rtcwake..."
    sudo apt-get update && sudo apt-get install -y util-linux
fi

# Define the Wake Up Job (Runs at 1:55 AM to set the wake alarm for 2:00 AM)
# Note: rtcwake needs to be set *before* the machine sleeps.
# A better approach for a server is to set the wake time right before shutdown.

# Define the Shutdown Job (Runs at 8:00 AM)
SHUTDOWN_CMD="sudo systemctl suspend" # Use suspend for faster wake, or 'poweroff' for deep sleep

# Create a script that handles the sleep logic + setting the next wake time
cat << 'EOF' > ~/personal/PhotoSynth/scripts/sleep_and_wake.sh
#!/bin/bash
# Set wake alarm for 2:00 AM tomorrow
sudo rtcwake -m no -l -t $(date +%s -d 'tomorrow 02:00')

# Suspend the system
echo "Suspending system... Wake set for 2:00 AM."
sudo systemctl suspend
EOF

chmod +x ~/personal/PhotoSynth/scripts/sleep_and_wake.sh

# Add to Crontab
# 1. At 8:00 AM, run the sleep script
(crontab -l 2>/dev/null; echo "0 8 * * * /home/adityadas/personal/PhotoSynth/scripts/sleep_and_wake.sh >> /tmp/photosynth_power.log 2>&1") | crontab -

echo "✅ Power schedule configured:"
echo "   - Sleep at: 8:00 AM (via cron)"
echo "   - Wake at:  2:00 AM (via rtcwake set during sleep)"
echo ""
echo "⚠️  IMPORTANT: Run 'sudo /home/adityadas/personal/PhotoSynth/scripts/sleep_and_wake.sh' manually once to start the cycle!"
