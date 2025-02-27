# Changelog

## [2.0.0] - 2025-02-27

### Added
- Sliding window (60 timesteps) for temporal predictions
- Multi-task learning for simultaneous predictions
- Parameter reduction through pruning and weight clipping
- Enhanced visualization with acc/val_acc and loss/val_loss plots
- Proper air quality categorization with 'Very Unhealthy' class

### Changed
#### Model Architecture
- Reduced CNN filters (32->64 instead of 64->128->256)
- Single efficient BiLSTM layer (32 units)
- Optimized dense layers
- Added multi-task outputs for temperature and humidity

#### Data Processing
- Fixed SMOTE implementation with 'not majority' strategy
- Proper handling of class distribution
- Added sliding window data preparation
- Updated air quality category boundaries

#### Training Process
- Combined classification and regression training
- Added visualization for all metrics
- Improved parameter efficiency
- Added model optimization techniques

### Fixed
- Data balancing approach
- Air quality categorization logic
- Parameter count reduction
- Training visualization
- Sequence length handling

## [1.0.0] - 2025-02-27

### Initial Release
- Basic hybrid model implementation
- Standard data processing
- Basic visualization