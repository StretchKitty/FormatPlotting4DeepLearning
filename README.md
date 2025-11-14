# PlotFormatter - Consistent Visualization for Deep Learning

A Python module for creating consistent, publication-ready plots with automatic color mapping and log-log loss plot.

## Features

- **Consistent Color Mapping**: Ensures the same variable always gets the same color across different plots
- **Automatic Scientific Notation**: All loss values displayed in scientific notation with proper LaTeX formatting
- **Multiple Data Format Support**: Works with lists, NumPy arrays, and PyTorch tensors
- **Logger Integration**: Seamless integration with LoggerFormatter for experiment tracking

## Installation

```bash
# Required dependencies
pip install numpy matplotlib seaborn tqdm

# Optional for PyTorch tensor support
pip install torch
```

## Quick Start

```python
from Plotter import PlotFormatter
import numpy as np

# Initialize the plotter
plotter = PlotFormatter()

# Create sample loss data
losses = {
    'train': [10 * np.exp(-0.05 * i) for i in range(100)],
    'test': [12 * np.exp(-0.04 * i) for i in range(100)]
}

# Create color map and plot
cmap = plotter.create_variable_cmap(losses)
plotter.plot_loss(losses, cmap)
```

## Step-by-Step Tutorial

### Step 1: Basic Loss Plotting

```python
from Plotter import PlotFormatter
import numpy as np

plotter = PlotFormatter()

# Single loss curve
single_loss = [10 * np.exp(-0.05 * i) + 0.1 for i in range(100)]
plotter.plot_loss({'loss': single_loss})
```

### Step 2: Multiple Loss Curves with Color Mapping

```python
# Create multiple loss curves
losses = {
    'SGD': [10 * np.exp(-0.02 * i) for i in range(100)],
    'Adam': [10 * np.exp(-0.05 * i) for i in range(100)],
    'RMSprop': [10 * np.exp(-0.04 * i) for i in range(100)]
}

# Generate color map
color_map = plotter.create_variable_cmap(losses)
print(f"Color assignments: {color_map}")

# Plot with consistent colors
plotter.plot_loss(losses, color_map)
```

### Step 3: Different Data Types

```python
import torch

# PlotFormatter handles different data types seamlessly
mixed_data = {
    'list_loss': [1.0, 0.8, 0.6, 0.4, 0.2],
    'numpy_loss': np.array([1.1, 0.9, 0.7, 0.5, 0.3]),
    'tensor_loss': torch.tensor([1.2, 1.0, 0.8, 0.6, 0.4])
}

cmap = plotter.create_variable_cmap(mixed_data)
plotter.plot_loss(mixed_data, cmap)
```

### Step 4: Custom Color Schemes

```python
# Use different matplotlib colormaps
losses = {
    'model_A': np.random.exponential(scale=0.1, size=50),
    'model_B': np.random.exponential(scale=0.2, size=50),
    'model_C': np.random.exponential(scale=0.15, size=50)
}

# Try different colormaps
cmap_viridis = plotter.create_variable_cmap(losses, cmap_name='viridis')
cmap_plasma = plotter.create_variable_cmap(losses, cmap_name='plasma')
cmap_tab10 = plotter.create_variable_cmap(losses, cmap_name='tab10')

plotter.plot_loss(losses, cmap_viridis)
plotter.plot_loss(losses, cmap_plasma)
plotter.plot_loss(losses, cmap_tab10)
```

## Advanced Example: Consistent Colors Across Multiple Plots

This example demonstrates how the same models maintain consistent colors across different experiments:

```python
import numpy as np
from Plotter import PlotFormatter

plotter = PlotFormatter()
np.random.seed(42)

# Define model names that will appear in multiple experiments
model_names = ['ResNet50', 'VGG16', 'MobileNet', 'EfficientNet']

# Experiment 1: Training on Dataset A
experiment_1 = {
    'ResNet50': [8 * np.exp(-0.06 * i) + 0.05 * np.random.randn() for i in range(100)],
    'VGG16': [9 * np.exp(-0.04 * i) + 0.08 * np.random.randn() for i in range(100)],
    'MobileNet': [7 * np.exp(-0.07 * i) + 0.06 * np.random.randn() for i in range(100)]
}

# Experiment 2: Training on Dataset B (different subset of models)
experiment_2 = {
    'ResNet50': [10 * np.exp(-0.05 * i) + 0.07 * np.random.randn() for i in range(150)],
    'EfficientNet': [6 * np.exp(-0.08 * i) + 0.04 * np.random.randn() for i in range(150)],
    'VGG16': [11 * np.exp(-0.03 * i) + 0.09 * np.random.randn() for i in range(150)]
}

# Experiment 3: Fine-tuning results
experiment_3 = {
    'MobileNet': [3 * np.exp(-0.09 * i) + 0.02 * np.random.randn() for i in range(50)],
    'EfficientNet': [4 * np.exp(-0.10 * i) + 0.03 * np.random.randn() for i in range(50)]
}

# Create a master color map for all models
all_models = {name: [] for name in model_names}
master_cmap = plotter.create_variable_cmap(all_models, cmap_name='tab10')

# Plot each experiment with consistent colors
print("Plotting Experiment 1: Dataset A")
plotter.plot_loss(experiment_1, master_cmap)

print("Plotting Experiment 2: Dataset B") 
plotter.plot_loss(experiment_2, master_cmap)

print("Plotting Experiment 3: Fine-tuning")
plotter.plot_loss(experiment_3, master_cmap)

# Notice: ResNet50 will have the same color in Exp 1 and 2
# VGG16 will have the same color in Exp 1 and 2
# MobileNet will have the same color in Exp 1 and 3
# EfficientNet will have the same color in Exp 2 and 3
```

## Integration with Logger

```python
from Plotter import PlotFormatter
from Logger import LoggerFormtter
import numpy as np

# Initialize both modules
plotter = PlotFormatter()
logger = LoggerFormtter(project_name="my_experiment")

# Create and plot losses
losses = {
    'train': np.random.exponential(scale=0.1, size=100),
    'validation': np.random.exponential(scale=0.15, size=100)
}

cmap = plotter.create_variable_cmap(losses)
plotter.plot_loss(losses, cmap, logger=logger, fig_name="training_curves.png")
```

## Output Format

The legend displays losses in scientific notation:
- Values are shown as: `model_name: min a.bc × 10^x at epoch y`
- Example: `Adam: min 2.34 × 10^{-4} at epoch 87`
- All values use scientific notation, including 10^-1, 10^0, and 10^1

## API Reference

### `PlotFormatter()`
Initialize the formatter (no parameters needed).

### `create_variable_cmap(variables, cmap_name='viridis')`
Create a consistent color mapping for variables.

**Parameters:**
- `variables` (Dict[str, Any]): Dictionary with plot names as keys and data as values
- `cmap_name` (str): Matplotlib colormap name (default: 'viridis')

**Returns:**
- Dict[str, Tuple]: Dictionary mapping variable names to RGBA color tuples

### `plot_loss(variables, cmap_dict=None, logger=None, fig_name=None, figsize=(10,6))`
Plot loss curves with log-log scale and scientific notation.

**Parameters:**
- `variables` (Dict[str, Any]): Dictionary of loss data to plot
- `cmap_dict` (Optional[Dict]): Color mapping from create_variable_cmap
- `logger` (Optional): LoggerFormatter instance for saving
- `fig_name` (Optional[str]): Filename for saving (requires logger)
- `figsize` (Tuple[int, int]): Figure dimensions

## Tips

1. **Color Consistency**: Always create a master color map when comparing models across multiple plots
2. **Colormap Selection**: Use 'tab10' or 'Set1' for ≤10 variables, 'hsv' or 'viridis' for more
3. **Performance**: The progress bar helps track plotting for large datasets
4. **Scientific Notation**: All values are automatically formatted with 2 significant figures
