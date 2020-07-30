# ccc-kernels
This repository contains computational kernels of the CCC code.

## topaz results (2x V100, PCIe3)
* single core: 113.420s
* 16 cores (OpenMP): 11.563s 
* 1 GPU (OpenACC): 1.831s
* 2 GPUs (OpenMP + OpenACC): 1.111s

## profiling
Significant ammount of time is spent on copying data from Host to Device 

            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   65.88%  1.49858s      1991  752.68us  1.5360us  1.3836ms  [CUDA memcpy HtoD]
                   16.71%  380.12ms       415  915.96us  889.37us  1.0600ms  MAIN__1F1L84_122_gpu
                   16.30%  370.78ms       415  893.46us  876.09us  916.02us  MAIN__1F1L84_130_gpu
                    0.65%  14.801ms       415  35.664us  35.008us  42.623us  MAIN__1F1L84_115_gpu
                    0.46%  10.536ms      1245  8.4630us  1.2480us  28.320us  [CUDA memcpy DtoH]
                    0.00%  3.8400us         2  1.9200us  1.8560us  1.9840us  [CUDA memset]

