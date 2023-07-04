# run_FctSeg_seed.py
import subprocess
import time
import pyautogui


def run_FctSeg_seed_with_variable(new_value):
    with open("FctSeg_seed.py", "r") as file:
        lines = file.readlines()

    # Modify the variable value in the FctSeg_seed
    for i, line in enumerate(lines):
        if "seg_out = Fseg(Ig, ws=" in line:
            lines[i] = f"   seg_out = Fseg(Ig, ws= {new_value}, seeds=seeds)\n"

    with open("test.py", "w") as file:
        file.writelines(lines)

def main():
    new_variable_value = 50  # The new value you want to set for my_variable
    run_FctSeg_seed_with_variable(new_variable_value)

    # Run the modified FctSeg_seed using subprocess
    subprocess.Popen(["python3", "FctSeg_seed.py"])

    # Wait for some time to ensure the window has opened completely
    time.sleep(5)

    # Take a screenshot
    screenshot = pyautogui.screenshot()

    # Save the screenshot
    screenshot.save(f"./results/ws={new_variable_value}.png")

if __name__ == "__main__":
    main()
