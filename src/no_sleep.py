import time
import ctypes

# Prevent sleep
ctypes.windll.kernel32.SetThreadExecutionState(0x80000002)

try:
    while True:
        time.sleep(60)
except KeyboardInterrupt:
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
