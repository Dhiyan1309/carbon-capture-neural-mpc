# AI-Driven Process Control & IT/OT Integration for Carbon Capture

This repository contains the complete Software-in-the-Loop (SIL) Digital Twin architecture bridging a high-level predictive LSTM neural network with deterministic industrial hardware (Siemens/CODESYS).

## 📌 Core Components
* **System Identification:** MATLAB FOPDT extraction for chemical thermal inertia.
* **The Brain:** Python-based LSTM neural network acting as a predictive digital twin.
* **The Optimizer:** Model Predictive Control (MPC) algorithm calculating mathematically optimal, feathered valve trajectories.
* **The Cyber-Bridge:** Modbus TCP pipeline translating floating-point math to industrial execution bytes.
* **The Muscle:** IEC 61131-3 CODESYS PLC enforcing strict `<10ms` hardware safety watchdogs.

## 🛠️ Execution Pipeline
1. Run `matlab_system_id/fopdt_simulation.m` to generate the raw thermal dataset.
2. Run `python_ai_controller/train_lstm.py` to build and save the AI model.
3. Boot the CODESYS Virtual PLC and load `codesys_plc_logic/Main_Routine.st`.
4. Run `python_ai_controller/run_mpc.py` to bridge the AI to the PLC and generate the final performance graph.
