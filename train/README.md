# train - Original training for the final detection model

From-scratch reproduction of the detection model training data
(RangeLDM redrawn batch B + d50k + pedestrian alignment v3) and the
VoxelRCNN training itself (GBlobs fork, 4 GPUs batch 16 / 40 epochs).
Includes data-prep/generation code, the official RangeLDM code, GBlobs training
code and the training config. Raw data: see `docs/TRAINING_0TO1.md`.
