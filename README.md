# Flight6DOF-Simulator

## Overview

**Flight6DOF-Simulator** is a **6-Degree-of-Freedom (6-DOF) flight dynamics simulator** built in **Python** using **Streamlit**.  
It models a light aircraft’s full rigid-body motion, including **aerodynamic forces**, **control inputs**, and **attitude dynamics**, with real-time visualizations of flight trajectory, airspeed, altitude, and orientation.

The project provides a clean web-based interface to interactively control throttle, elevator, aileron, and rudder — ideal for **aerospace education**, **control design**, or **simulation prototyping**.

---

## Features

- **6-DOF Aircraft Model** — Simulates both translational and rotational motion.  
- **Aerodynamic Force & Moment Modeling** — Includes lift, drag, and control effects.  
- **Interactive Controls** — Adjust flight surfaces and engine throttle in real time.  
- **Dynamic Visualization** — 2D flight metrics and 3D trajectory plots.  
- **Parameter Customization** — Easily modify aircraft properties for testing.  
- **Streamlit Web UI** — Runs entirely in your browser with no extra setup.  

---

## Requirements

Make sure you have **Python 3.8+** installed.

Install dependencies with:

```bash
pip install -r requirements.txt
````

If you don’t have a `requirements.txt` file yet, create one with the following content:

```text
streamlit
numpy
scipy
matplotlib
```

---

## Installation and Running

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/Flight6DOF-Simulator.git
   cd Flight6DOF-Simulator
   ```

2. **Install required packages:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the simulator:**

   ```bash
   streamlit run flight6dof.py
   ```

4. **Access the web interface:**
   Open your browser and go to:

   ```
   http://localhost:8501
   ```

---

## Example Output

* Real-time graphs of:

  * Altitude vs. Time
  * Airspeed vs. Time
  * Roll and Pitch angles
* 3D visualization of aircraft trajectory and attitude.
* Adjustable controls for throttle, elevator, aileron, and rudder.

---

## Tech Stack

* **Python 3**
* **Streamlit** — Interactive UI
* **NumPy** — Vector math and computation
* **SciPy** — Quaternion and rotation transformations
* **Matplotlib** — Real-time graphing and 3D visualization

---

## License

Open-source under the **MIT License** — contributions and improvements are welcome.

```
```
