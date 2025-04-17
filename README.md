# KMeans Parallelization: Serial, OpenMP, and CUDA

This project implements **KMeans clustering** in three versions:  
🟢 Serial (C), 🟡 OpenMP (multi-threaded CPU), 🔵 CUDA (GPU).  
Designed for **high-performance computing coursework (CSCI 5451 @ UMN)** and extended for benchmarking and visualization.

---

## 🧠 Project Overview

This project performs **unsupervised learning on MNIST digit images**, clustering data points by pixel features into `k` clusters using the KMeans algorithm.

Each version was designed to progressively improve runtime:
- `kmeans_serial.c` – Single-threaded C implementation
- `kmeans_omp.c` – Multi-threaded version using OpenMP
- `kmeans_cuda.cu` – GPU-accelerated version using CUDA

Each implementation:
- Reads MNIST-like feature data from a plain-text file
- Iteratively updates cluster centers and assignments
- Tracks convergence via assignment changes
- Outputs cluster center images (`.pgm`) and a confusion matrix

This project was originally based on [Assignment 3 of UMN CSCI 5451](https://www-users.cse.umn.edu/~kauffman/5451/a3.html), which specified the core algorithm and dataset format.

---

## 📁 Repository Structure

```
kmeans-parallel/
├── kmeans_serial.c           # Serial version
├── kmeans_omp.c              # OpenMP version
├── kmeans_cuda.cu            # CUDA version
├── kmeans_util.c/cu/h        # Shared utility code
├── Makefile                  # Easy compilation
├── scripts/                  # Benchmarking scripts
│   ├── kmeans-omp-time.sh
│   └── kmeans-cuda-time.sh
├── results/                  # Sample output (txt/pgm)
├── test/                     # Org-mode test runs
├── mnist-data/               # Data files or download instructions
├── kmeans.py                 # Optional Python reference implementation
└── README.md
```

---

## 🛠️ Build

### ✅ Using the Makefile
```bash
make
```

### 🔧 Or compile manually:

```bash
gcc -O3 kmeans_serial.c kmeans_util.c -o kmeans_serial -lm

gcc -O3 -fopenmp kmeans_omp.c kmeans_util.c -o kmeans_omp -lm

nvcc -O3 kmeans_cuda.cu kmeans_util.cu -o kmeans_cuda
```

---

## ▶️ Run

Each binary takes the following arguments:
```bash
./<binary> <datafile> <nclust> [savedir] [maxiter]
```

### Example:
```bash
./kmeans_serial mnist-data/digits_all_5e3.txt 20 results_serial 100
./kmeans_omp mnist-data/digits_all_5e3.txt 20 results_omp 100
./kmeans_cuda mnist-data/digits_all_5e3.txt 20 results_cuda 100
```

---

## 📊 Benchmarking

### CUDA:
```bash
bash scripts/kmeans-cuda-time.sh
```

### OpenMP:
```bash
bash scripts/kmeans-omp-time.sh
```

These scripts:
- Test on varying dataset sizes (5k, 10k, 30k MNIST samples)
- Iterate over OpenMP thread counts (1–32)
- Log timing output using `/usr/bin/time`

---

## 📷 Output Samples

- `results/*.txt` – runtime logs, confusion matrix, convergence
- `results/*.pgm` – grayscale visualizations of cluster centers
- `labels.txt` – original and predicted labels for inspection

---

## 📘 Dataset Format (from A3 instructions)

Each line of the dataset:
```
[label] : <f1> <f2> ... <f784>
```
- `label` is a ground-truth digit [0–9]
- `:` is ignored
- `f1..f784` are flattened grayscale pixel intensities (28x28)

You may generate your own data using MNIST or use sample files like:
- `digits_all_5e3.txt`
- `digits_all_1e4.txt`

---

## 💡 Key Learning Outcomes

- Implemented KMeans from scratch in C and CUDA
- Compared performance across CPU and GPU environments
- Used OpenMP's thread reduction and critical sections
- Optimized CUDA memory access patterns
- Benchmarked convergence rates and runtime across versions

---

## 📎 Contact

Made with ❤️ by [Your Name](https://your-portfolio.com)  
Inspired by [CSCI 5451: High Performance Computing](https://www-users.cse.umn.edu/~kauffman/5451/)
