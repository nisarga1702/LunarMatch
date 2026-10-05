"""
Keep-Alive Script for Render Backend Server (LunarMatch-TGS).
Pings the Render app every 10 minutes to prevent free-tier idling/sleeping.
"""

import urllib.request
import time
import sys
import os
from datetime import datetime

# Default Render app URL (can be overridden via command line argument or RENDER_URL env var)
DEFAULT_URL = os.getenv("RENDER_URL", "https://lunarmatch-tgs.onrender.com")

def ping_server(url):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "LunarMatch-KeepAlive/1.0"}
        )
        with urllib.request.urlopen(req, timeout=15) as response:
            status = response.getcode()
            print(f"[{timestamp}] SUCCESS | Pinged {url} | Status Code: {status}")
            return True
    except Exception as e:
        print(f"[{timestamp}] WARNING | Failed to ping {url} | Error: {e}")
        return False

def run_keep_alive(url=DEFAULT_URL, interval_seconds=600):
    print("=" * 65)
    print(f"🚀 LunarMatch Keep-Alive Service Started")
    print(f"📍 Target Server URL: {url}")
    print(f"⏱️  Ping Interval: Every {interval_seconds // 60} minutes ({interval_seconds} seconds)")
    print("=" * 65)
    
    # Initial immediate ping
    ping_server(url)
    
    while True:
        time.sleep(interval_seconds)
        ping_server(url)

if __name__ == "__main__":
    target_url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL
    # Allow custom interval via second arg
    interval = int(sys.argv[2]) if len(sys.argv) > 2 else 600
    
    try:
        run_keep_alive(target_url, interval)
    except KeyboardInterrupt:
        print("\n👋 Keep-Alive service stopped by user.")
