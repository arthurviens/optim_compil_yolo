# Optimizing YOLO with a Deep Learning Compiler

This repository offers some initial documentation, references and tools in preparation for an internship at the centre de recherche en informatique (CRI) Mines Paris - PSL.

## References
### YOLO Architectures

[1] - Terven, J., Córdova-Esparza, D. M., & Romero-González, J. A. (2023). A comprehensive review of yolo architectures in computer vision: From yolov1 to yolov8 and yolo-nas. Machine learning and knowledge extraction, 5(4), 1680-1716.

[2] - Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). You only look once: Unified, real-time object detection. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 779-788).

[3] - Redmon, J., & Farhadi, A. (2017). YOLO9000: better, faster, stronger. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 7263-7271).

[4] - Redmon, J., & Farhadi, A. (2018). Yolov3: An incremental improvement. arXiv preprint arXiv:1804.02767.

[5] - Bochkovskiy, A., Wang, C. Y., & Liao, H. Y. M. (2020). Yolov4: Optimal speed and accuracy of object detection. arXiv preprint arXiv:2004.10934.

### Deep Learning Compilers
[6] - Li, M., Liu, Y., Liu, X., Sun, Q., You, X., Yang, H., ... & Qian, D. (2020). The deep learning compiler: A comprehensive survey. IEEE Transactions on Parallel and Distributed Systems, 32(3), 708-727.

[7] - Ragan-Kelley, J., Barnes, C., Adams, A., Paris, S., Durand, F., & Amarasinghe, S. (2013). Halide: a language and compiler for optimizing parallelism, locality, and recomputation in image processing pipelines. Acm Sigplan Notices, 48(6), 519-530. <br/>
+ See video below

[8] - Chen, T., Moreau, T., Jiang, Z., Zheng, L., Yan, E., Shen, H., ... & Krishnamurthy, A. (2018). {TVM}: An automated {End-to-End} optimizing compiler for deep learning. In 13th USENIX Symposium on Operating Systems Design and Implementation (OSDI 18) (pp. 578-594).

[9] - Roesch, J., Lyubomirsky, S., Weber, L., Pollock, J., Kirisame, M., Chen, T., & Tatlock, Z. (2018, June). Relay: A new ir for machine learning frameworks. In Proceedings of the 2nd ACM SIGPLAN international workshop on machine learning and programming languages (pp. 58-68).

[10] - Feng, S., Hou, B., Jin, H., Lin, W., Shao, J., Lai, R., ... & Chen, T. (2023, January). Tensorir: An abstraction for automatic tensorized program optimization. In Proceedings of the 28th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2 (pp. 804-817).

[11] - Lai, R., Shao, J., Feng, S., Lyubomirsky, S. S., Hou, B., Lin, W., ... & Chen, T. (2023). Relax: composable abstractions for end-to-end dynamic machine learning. arXiv preprint arXiv:2311.02103.

### Optimizations

[12] - Jia, Z., Padon, O., Thomas, J., Warszawski, T., Zaharia, M., & Aiken, A. (2019, October). TASO: optimizing deep learning computation with automatic generation of graph substitutions. In Proceedings of the 27th ACM Symposium on Operating Systems Principles (pp. 47-62).

[13] - Unger, C., Jia, Z., Wu, W., Lin, S., Baines, M., Narvaez, C. E. Q., ... & Aiken, A. (2022). Unity: Accelerating {DNN} training through joint optimization of algebraic transformations and parallelization. In 16th USENIX Symposium on Operating Systems Design and Implementation (OSDI 22) (pp. 267-284).

[14] - Niu, W., Guan, J., Wang, Y., Agrawal, G., & Ren, B. (2021, June). Dnnfusion: accelerating deep neural networks execution with advanced operator fusion. In Proceedings of the 42nd ACM SIGPLAN International Conference on Programming Language Design and Implementation (pp. 883-898).

[15] - Qiao, B., Reiche, O., Hannig, F., & Teich, J. (2019, February). From loop fusion to kernel fusion: A domain-specific approach to locality optimization. In 2019 IEEE/ACM International Symposium on Code Generation and Optimization (CGO) (pp. 242-253). IEEE.

[16] - Cai, X., Wang, Y., & Zhang, L. (2022). Optimus: An operator fusion framework for deep neural networks. ACM Transactions on Embedded Computing Systems, 22(1), 1-26.

### Memory optimizations
[17] - Levental, M. (2022). Memory planning for deep neural networks. arXiv preprint arXiv:2203.00448.

[18] - Artemev, A., An, Y., Roeder, T., & van der Wilk, M. (2022). Memory safe computations with XLA compiler. Advances in Neural Information Processing Systems, 35, 18970-18982.

[19] - Pisarchyk, Y., & Lee, J. (2020). Efficient memory management for deep neural net inference. arXiv preprint arXiv:2001.03288.


## Other learning material
- An interesting description of optimization and Halide : https://www.youtube.com/watch?v=1ir_nEfKQ7A&t=126s
- A visualisation of tiling effects on matrix multiplication
    - Naive implementation https://www.youtube.com/watch?v=QYpH-847z0E
    - Tiled implementation, B transposed https://www.youtube.com/watch?v=aMvCEEBIBto
    - Visible L1 cache https://www.youtube.com/watch?v=aU1zsFk36l0
    - 2 level tiled https://www.youtube.com/watch?v=3XfHL6nlB08

- Interesting blog on optimization/compilation in AI : https://www.aussieai.com/research/list, notably [this page](https://www.aussieai.com/research/compilers)

Interesting optimizations:
* Graph optimizations : [12, 13]
* Kernel fusion (aka operator fusion) : [14, 15, 16]
* Loop fusion/fission
* Tiling
* Vectorization
* Unrolling
* Memory planning

## Tools
### Apache TVM : https://tvm.apache.org/ <br/>
Installation https://tvm.apache.org/docs/install/index.html
- Installation on Linux is very highly recommanded (possibly Docker)
- If on windows, installation on Docker is highly recommanded. Else, good luck !
- If targeting GPU, install CUDA : https://developer.nvidia.com/cuda-12-4-0-download-archive/?target_os=Linux (do not hesitate to follow the considered section of [NVIDIA install Documentation](https://docs.nvidia.com/cuda/cuda-installation-guide-linu))
- For profiling, [building TVM with PAPI support](https://tvm.apache.org/docs/v0.8.0/how_to/profile/papi.html) is recommended.

Official quickstart tutorial https://mlc.ai/docs/get_started/tutorials/quick_start.html <br/>
Quality tutorial : https://mlc.ai/chapter_introduction/index.html <br/>

### Torch
https://pytorch.org/

If we want to to compare with Torch:
- Torch Script : https://pytorch.org/docs/stable/jit.html
- Torch Profiler : https://pytorch.org/docs/stable/profiler.html
- Torch compile : https://pytorch.org/docs/stable/torch.compiler.html

### ONNX 
ONNX is a deep learning model format, with frontends to most deep learning frameworks and compilers https://onnx.ai/
* Onnx model zoo : https://github.com/onnx/models
    * MNIST CNN model : https://github.com/onnx/models/tree/main/validated/vision/classification/mnist
    * TinyYOLOv2 : https://github.com/onnx/models/tree/main/validated/vision/object_detection_segmentation/tiny-yolov2


### Netron
Visualisation of an ONNX model graph https://netron.app/


## Internship approximative planning
Approximative planning to have vision over the next weeks, **however**, you're completely free to start experimenting right away if you feel like it.

* Weeks 1 & 2 : Learning the context, reading bibliography, install the environment and the tools.
* Weeks 3 & 4 : Experiments regarding computation time optimization
* Weeks 5 & 6 : Experiments regarding memory optimization (peak memory consumption)
* Weeks 7 & 8 : Analysis of the experiments results, show and explain relevant optimizations, report redaction.