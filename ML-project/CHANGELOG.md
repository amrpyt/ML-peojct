# Changelog

## [1.1.0] - 2025-02-27

### Added
- New improved CNN-BiLSTM hybrid architecture
- Training history visualization in notebooks
- Comprehensive model comparison analysis
- Advanced training callbacks configuration

### Changed
#### CNN Architecture Improvements
- Increased filter sizes (64->128->256)
- Added padding='same' to maintain sequence dimensions
- Added BatchNormalization after each Conv1D layer
- Optimized pooling layer placement
- Added proper sequence length handling

#### BiLSTM Improvements
- Added dual BiLSTM layers (128 and 64 units)
- Implemented proper sequence handling between CNN and BiLSTM
- Added dropout (0.3) between BiLSTM layers
- Improved temporal feature processing

#### Dense Layer Optimization
- Increased layer sizes (128->64)
- Added BatchNormalization for better training stability
- Optimized dropout rates for better generalization

#### Training Process Enhancements
- Added ReduceLROnPlateau callback for adaptive learning
- Increased early stopping patience to 10 epochs
- Added validation monitoring during training
- Improved batch size handling

### Fixed
- Negative dimension error in CNN layers
- Sequence length preservation issues
- Model architecture connectivity problems
- Training stability issues

## [1.0.0] - 2025-02-27

### Initial Release
- Basic CNN-BiLSTM implementation
- Standard training process
- Basic model evaluation