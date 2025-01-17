# Project Description

This project involves a Reinforcement Learning model designed to play the Super Mario Bros v1 game! The model uses RL with DDQN and the PyTorch library to train an agent to navigate through the game environment effectively. 

## Initial Implementation -- On MacBook M2 Chip

Initially, the project was set up to run on a MacBook M2 chip, utilizing the MPS backend for computations. This setup was constrained to the single-GPU capabilities of the MacBook, focusing on efficient but limited resource usage. The training and model execution were done entirely on the CPU or the integrated GPU of the M2 chip, which allowed for initial testing and development without the need for more complex hardware configurations. This took about a week to train, especially because I kept `DISPLAY` to true....whoops!

## Using 2 GPUs -- Trying to Use Two GPUs (Work in progress)

With the aim of enhancing performance and reducing training time, the project was adapted to utilize two GPUs. Key changes include setting up the model to support CUDA, enabling it to leverage the parallel processing power of modern GPUs. This was achieved by integrating `torch.nn.DataParallel`, which allows the model to distribute its operations across multiple GPUs, effectively parallelizing the computation and data handling. To specify here, our model is small enough to fit on one GPU and thus, we only need to parallelize the data and episodes...not the model itself. Modifications were made to ensure that tensors are correctly moved to and processed on the GPUs, and that model parameters are synchronized across GPUs during training. These changes help in speeding up the learning process and in handling more complex or larger-scale training scenarios. Key modifications are:
* Implemented PyTorch's `DistributedDataParallel` for efficient multi-GPU training
* Used NCCL (NVIDIA Collective Communications Library) backend for optimal GPU-to-GPU communication
* Set up process groups using `torch.distributed` with a dedicated process per GPU
* Implemented proper process initialization with rank-based device assignment
* Maintained separate experience replay buffers per GPU to reduce memory transfer overhead
* Implemented gradient synchronization across GPUs using `dist.all_reduce`
* Utilized device-specific tensor allocation for efficient memory usage
* Implemented rank-based conditionals for display and model saving operations

### Model Distribution

* Each GPU maintains its own copy of: Online network (for current policy); Target network (for stable Q-value estimation); Experience replay buffer

* Gradients are synchronized during backpropagation to ensure consistent learning
Model saving is handled by the main process (rank 0) to prevent conflicts

However, this is still being tested and has not been timed against the first implementation yet!

## Next: Potential Fixes

- **Memory Management**: Monitor and optimize GPU memory usage to prevent out-of-memory errors, potentially adjusting batch sizes.
- **Batch Size Tuning**: Experiment with different batch sizes to find the optimal setting for the new GPU configuration, which can further enhance training efficiency.

