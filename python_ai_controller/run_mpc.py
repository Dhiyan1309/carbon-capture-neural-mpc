import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from scipy.optimize import minimize

from pymodbus.client import ModbusTcpClient

print("Connecting to CODESYS Virtual PLC...")
plc = ModbusTcpClient('127.0.0.1', port=502)
if plc.connect():
    print("SUCCESS: Industrial Cyber-Bridge Established!")
else:
    print("WARNING: Could not connect to PLC. Check if CODESYS is running.")
plt.clf()
print("1. Waking up the AI Brain...")
# Load the trained LSTM from Phase 2
model = load_model(r'C:\Users\dhiya\OneDrive\Desktop\CCS_proj\ccs_lstm_brain.keras')

TARGET_TEMP_SCALED = 0.6  # Our goal temperature (scaled between 0 and 1)
Q = 100.0                 # Penalty for missing the target temperature
R = 50.0

# --- SIMULATION SETUP ---
sequence_length = 60
simulation_steps = 100

# We need 60 minutes of history to kickstart the LSTM. 
# Let's assume the valve has been sitting at 50% (0.5 scaled) for the last hour.
past_valve_commands = [0.5] * sequence_length 

# Arrays to store data for our final graph
history_valve = []
history_temp = []

print("2. Starting the MPC Closed-Loop Simulation...")

# The Cost Function for the Optimizer
def mpc_cost_function(proposed_valve, current_history):
    # 1. Create a temporary history array with the new proposed valve move
    test_history = current_history[1:] + [proposed_valve[0]]
    test_input = np.array(test_history).reshape(1, sequence_length, 1)
    
    # 2. Ask the LSTM: "If I make this valve move, what happens?"
    predicted_temp = model.predict(test_input, verbose=0)[0][0]
    
    # 3. Calculate the Cost (J)
    temp_error = (predicted_temp - TARGET_TEMP_SCALED) ** 2
    valve_movement = (proposed_valve[0] - current_history[-1]) ** 2
    
    cost = (Q * temp_error) + (R * valve_movement)
    return cost

# Run the plant for 100 minutes
for step in range(simulation_steps):
    
    # 1. The Optimizer hunts for the best valve position (between 0% and 100%)
    # We give it the current valve position as its starting guess
    current_valve = past_valve_commands[-1]
    
    result = minimize(
        mpc_cost_function, 
        x0=[current_valve], 
        args=(past_valve_commands,), 
        bounds=[(0.0, 1.0)], 
        method='L-BFGS-B',       # A smarter optimizer for Neural Networks
        options={'eps': 0.05}    # FORCE it to test a 5% valve jump to wake the AI up!
    )
    
    # 2. The Optimizer found the perfect move. Extract it.
    best_valve_move = result.x[0]
    
    # Convert the 0.0-1.0 AI value to a 0-100 integer for the PLC memory
    plc_command = int(best_valve_move * 100)
    
    # Fire the command over the network to CODESYS Modbus Register 0
    plc.write_register(0, plc_command)
    print(f"Network Send: {plc_command}% to Industrial Controller")
    
    # 3. Ask the LSTM what the actual temperature will be with this perfect move
    # (In real life, this is where we send the command to the Siemens PLC)
    new_history = past_valve_commands[1:] + [best_valve_move]
    new_input = np.array(new_history).reshape(1, sequence_length, 1)
    resulting_temp = model.predict(new_input, verbose=0)[0][0]
    
    # 4. Update our history for the next minute
    past_valve_commands.append(best_valve_move)
    past_valve_commands.pop(0) # Remove the oldest data point to keep it at 60
    
    # 5. Save data for graphing
    history_valve.append(best_valve_move * 100) # Convert back to 0-100%
    history_temp.append(resulting_temp)
    
    print(f"Step {step+1}/{simulation_steps} | Valve Cmd: {best_valve_move*100:.1f}% | Temp: {resulting_temp:.3f}")

print("3. Simulation Complete! Plotting Results...")

# --- PLOT THE RESULTS ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Top Graph: The Temperature hitting the target
ax1.plot(history_temp, 'b-', linewidth=2, label='AI Controlled Temperature')
ax1.axhline(TARGET_TEMP_SCALED, color='k', linestyle='--', label='Target Setpoint')
ax1.set_ylabel('Scaled Temperature')
ax1.set_title('Neural-MPC Controlling Carbon Capture Plant')
ax1.grid(True)
ax1.legend()

# Bottom Graph: What the AI did to the valve
ax2.step(range(simulation_steps), history_valve, 'r-', linewidth=2, label='MPC Valve Command')
ax2.set_ylabel('Valve Position (%)')
ax2.set_xlabel('Time (Minutes)')
ax2.grid(True)
ax2.legend()

import os

# 1. We will bypass OneDrive and save it to a guaranteed public folder
save_path = r'C:\Users\Public\mpc_FINAL_AI_GRAPH.png'

# 2. Save the high-res image
plt.tight_layout()
plt.savefig(save_path, dpi=300)
print(f"SUCCESS: Graph saved to the public drive: {save_path}")

# 3. FORCE Windows to instantly open the image on your screen
try:
    os.startfile(save_path)
except Exception as e:
    print(f"Could not force-open the image: {e}")

# 4. Also pop open the interactive Python window just in case
plt.show()

