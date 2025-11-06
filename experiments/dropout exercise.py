import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def study_loop(days):
    for day in range(1, days+1):
        print(f"\nDay {day} Study Schedule:")
        hour = 0
        while hour < 3:
            if hour == 1:
                print("  > Take a break.")  
        hour += 1
        continue
    for session in range(2):
       if session == 0 and day % 2 == 0:
           print(f"    Hour {hour}: Deep learning practice!")
       else:
           print(f"    Hour {hour}: Review and summary.")
           hour += 1
       if hour > 3:
           print("  Warning: too much study!")
           break
    
    print("End of day", day)
print("Finish all study days.")

# 建議測試
# study_loop(2)

