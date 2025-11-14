import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatter
import matplotlib.cm as cm
from typing import Dict, Union, Optional, Tuple, Any
from tqdm import tqdm

# Hyperparameters
DEFAULT_FIGSIZE = (10, 6)

class PlotFormatter:
   """
   A class for formatting plots with consistent color mapping and loss visualization
   """
   
   def __init__(self):
       """
       Initialize PlotFormatter without additional functionality
       """
       pass
   
   def create_variable_cmap(self, variables: Dict[str, Any], cmap_name: str = 'viridis') -> Dict[str, Tuple[float, float, float, float]]:
       """
       Create a color map for variables to ensure consistent colors across plots
       
       Args:
           variables: Dictionary with plot names as keys and variables as values
           cmap_name: Name of matplotlib colormap to use, default 'viridis'
           
       Returns:
           Dictionary with same keys as input but color tuples as values
       """
       var_names = list(variables.keys())
       n_colors = len(var_names)
       
       cmap = cm.get_cmap(cmap_name)
       colors = [cmap(i / max(n_colors - 1, 1)) for i in range(n_colors)]
       
       color_map = {}
       for i, var_name in enumerate(tqdm(var_names, desc="Creating color map")):
           color_map[var_name] = colors[i]
       
       return color_map
   
   def plot_loss(self, 
                 variables: Dict[str, Any],
                 cmap_dict: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
                 logger=None,
                 fig_name: Optional[str] = None,
                 figsize: Tuple[int, int] = DEFAULT_FIGSIZE) -> None:
       """
       Plot training and/or test losses with log-log scale using color mapping
       
       Args:
           variables: Dictionary with plot names as keys and loss values as values
           cmap_dict: Dictionary with plot names as keys and color tuples as values
           logger: Logger instance for saving figures
           fig_name: Name for saved figure file
           figsize: Figure size tuple
       """
       plt.figure(figsize=figsize, dpi=300)
       
       if cmap_dict is None:
           cmap_dict = self.create_variable_cmap(variables)
       
       legend_entries = []
       for var_name, loss_values in tqdm(variables.items(), desc="Plotting losses"):
           if hasattr(loss_values, 'tolist'):
               loss_values = loss_values.tolist()
           elif not isinstance(loss_values, list):
               loss_values = list(loss_values)
           
           epochs = np.arange(1, len(loss_values) + 1)
           
           color = cmap_dict.get(var_name, (0, 0, 0, 1))
           plt.loglog(epochs, loss_values, label=var_name, linewidth=2, color=color)
           
           min_val = min(loss_values)
           min_epoch = loss_values.index(min_val) + 1
           
           if min_val == 0:
               formatted_val = "0"
           else:
               exponent = int(np.floor(np.log10(abs(min_val))))
               mantissa = min_val / (10 ** exponent)
               mantissa_str = f"{mantissa:.2f}".rstrip('0').rstrip('.')
               formatted_val = f"${mantissa_str} \\times 10^{{{exponent}}}$"
           
           legend_entry = f"{var_name}: min {formatted_val} at epoch {min_epoch}"
           legend_entries.append(legend_entry)
       
       plt.xlabel("Epoch")
       plt.ylabel("Loss")
       plt.grid(True, which="both", ls="-", alpha=0.2)
       
       handles, _ = plt.gca().get_legend_handles_labels()
       plt.legend(handles, legend_entries, loc='upper right')
       
       plt.tight_layout()
       
       if logger and fig_name:
           logger.save_fig(fig_name)
       else:
           plt.show()
