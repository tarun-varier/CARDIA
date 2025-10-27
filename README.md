# CARDIA
### Parallelized ECG Signal Analysis Pipeline

This project provides a C++ implementation of the **Pan-Tompkins algorithm** for detecting QRS complexes (heartbeats) in ECG signals. It includes two versions:
* `cardia_serial.cpp`: A standard, single-threaded implementation.
* `cardia_parallel.cpp`: An optimized, multi-threaded version using **OpenMP** for significant performance gains on multi-core processors.

The pipeline processes raw ECG data from a CSV file to identify R-peaks, calculate the average heart rate (BPM), and measure execution time.

<details>
<summary>About the Code ℹ️</summary>

The program implements the core stages of the Pan-Tompkins algorithm to transform the raw ECG signal into a feature waveform where peaks are easily detectable.

### Key Algorithm Steps
1.  **Bandpass Filtering**: A simple moving-average-based high-pass and low-pass filter is applied to remove baseline wander and high-frequency noise, isolating the QRS frequency band (~5-15 Hz).
2.  **Derivative**: Calculates the signal's derivative to highlight the steep slopes characteristic of the QRS complex.
3.  **Squaring**: Squares the derivative output point-by-point. This enhances the signal, makes all values positive, and further emphasizes the QRS peaks.
4.  **Moving Window Integration**: A moving window integrator generates a feature waveform by summing the energy of the squared signal over a defined window.
5.  **Peak Detection**: An adaptive thresholding mechanism is applied to the integrated signal to identify R-peaks while enforcing a refractory period to avoid multiple detections for a single beat.

### Parallelization & Optimizations (`cardia_parallel.cpp`)
The parallel version incorporates several key improvements:
* **OpenMP Parallelism**: Uses `#pragma omp` directives to distribute element-wise computations (filtering, squaring, statistics) across multiple CPU cores.
* **Loop Fusion**: The **derivative** and **squaring** steps are fused into a single parallel loop. This reduces memory overhead by eliminating an entire intermediate vector and minimizes the overhead of creating parallel regions.
* **Efficient CSV Loading**: The file reader is optimized by pre-reserving vector memory and using `strtod` for faster string-to-double conversion, avoiding the overhead of `stringstream`.

</details>

**Project Structure:**

```bash
Directory structure:
└── tarun-varier-cardia/
    ├── README.md
    ├── cardia_parallel.cpp
    └── cardia_serial.cpp
```


## How to Run

### Prerequisites
* A C++ compiler that supports OpenMP, such as **GCC (g++)**.

### Compilation
Open your terminal and use the following commands to compile the programs. The `-O3` flag enables aggressive compiler optimizations, and `-fopenmp` is required to link the OpenMP library for the parallel version.

**1. Compile the Serial Version:**
```bash
g++ cardia_serial.cpp -o cardia_serial
```
**2. Compile the Parallel Version:**
```bash
g++ cardia_parallel.cpp -o cardia_parallel
```

**Run the Code**
Run the compiled executable from the command line, providing the path to the input ECG data file as an argument.

```bash
./<executable_name> <path_to_your_ecg_data.csv>
```
