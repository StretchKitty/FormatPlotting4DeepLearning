import os
import json
import glob
import shutil
from datetime import datetime
from pathlib import Path
import types
import inspect
import sys
import random
import pickle

class LoggerFormtter:
    """
    日志管理器，用于创建实验日志目录和保存超参数
    """
    
    def __init__(self, project_name="exp", base_path="./logs"):
        """
        初始化Logger，创建日志目录
        
        Args:
            project_name: 项目名称，默认为"exp"
            base_path: 日志保存的基础路径，默认为"./logs"
        """
        self.project_name = project_name
        self.base_path = Path(base_path)
        
        self.base_path.mkdir(exist_ok=True)
        
        self._create_log_directory()
        
        self._save_source_file()
    
    def _create_log_directory(self):
        """
        创建带时间戳的日志目录
        """
        self.timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.log_dir = self.base_path / f"{self.timestamp}-{self.project_name}"
        self.log_dir.mkdir(exist_ok=True)
        
        print(f"Logger initialized with:")
        print(f"  Project name: {self.project_name}")
        print(f"  Base path: {self.base_path}")
        print(f"  Log directory: {self.log_dir}")
    
    def _save_source_file(self):
        """
        保存当前运行的源文件
        """
        try:
            if 'ipykernel' in sys.modules:
                try:
                    import ipynbname
                    nb_path = ipynbname.path()
                    dest_path = self.log_dir / nb_path.name
                    shutil.copy2(nb_path, dest_path)
                    print(f"Notebook saved to: {dest_path}")
                except:
                    print("Running in Jupyter but cannot save notebook file (ipynbname not available)")
            else:
                # 获取调用栈，向上查找直到找到非Logger.py的文件
                frame = inspect.currentframe()
                caller_file = None
                
                # 遍历调用栈
                while frame:
                    filename = frame.f_code.co_filename
                    if filename and not filename.endswith('Logger.py'):
                        # 找到第一个不是Logger.py的文件
                        source_file = Path(filename)
                        if source_file.exists() and source_file.suffix == '.py':
                            caller_file = source_file
                            break
                    frame = frame.f_back
                
                if caller_file:
                    dest_path = self.log_dir / caller_file.name
                    shutil.copy2(caller_file, dest_path)
                    print(f"Python file saved to: {dest_path}")
                else:
                    print("Could not determine source file to save")
                    
        except Exception as e:
            print(f"Could not save source file: {e}")

    
    def seed_everything(self, seed=42):
        """
        设置所有随机数生成器的种子
        
        Args:
            seed: 随机种子，默认为42
        """
        random.seed(seed)
        
        try:
            import numpy as np
            np.random.seed(seed)
        except ImportError:
            pass
        
        try:
            import torch
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except ImportError:
            pass
        
        print(f"Random seed set to: {seed}")
        return seed
    
    def save_fig(self, fig_name, dpi=300):
        """
        保存matplotlib图像到日志目录
        
        Args:
            fig_name: 图像文件名，必须提供
            dpi: 图像分辨率，默认300，不允许低于300
        """
        if not fig_name:
            raise ValueError("Figure name must be provided")
        
        if dpi < 300:
            raise ValueError(f"DPI must be at least 300, got {dpi}")
        
        try:
            import matplotlib.pyplot as plt
            
            if not fig_name.endswith(('.png', '.jpg', '.jpeg', '.pdf', '.svg')):
                fig_name += '.png'
            
            fig_name = f"{self.timestamp}-{self.project_name}-" + fig_name
            save_path = self.log_dir / fig_name
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
            return save_path
        except ImportError:
            raise ImportError("matplotlib is required for save_fig")
        except Exception as e:
            raise Exception(f"Failed to save figure: {e}")
    
    def save_large_vars(self, var_obj, var_name=None):
        """
        保存大变量到large_vars文件夹
        
        Args:
            var_obj: 要保存的变量对象
            var_name: 变量名称，如果不提供则尝试自动获取
        """
        large_vars_dir = self.log_dir / "large_vars"
        large_vars_dir.mkdir(exist_ok=True)
        
        if var_name is None:
            frame = inspect.currentframe()
            if frame and frame.f_back:
                caller_locals = frame.f_back.f_locals
                caller_globals = frame.f_back.f_globals
                
                for name, obj in {**caller_locals, **caller_globals}.items():
                    if obj is var_obj and not name.startswith('_'):
                        var_name = name
                        break
                
                if var_name is None:
                    var_name = f"unnamed_var_{datetime.now().strftime('%H%M%S')}"
                    print(f"Could not determine variable name, using: {var_name}")
        
        save_path = large_vars_dir / f"{var_name}.pkl"
        
        try:
            with open(save_path, 'wb') as f:
                pickle.dump(var_obj, f)
            
            file_size = save_path.stat().st_size / (1024 * 1024)
            print(f"Variable '{var_name}' saved to: {save_path}")
            print(f"  File size: {file_size:.2f} MB")
            return save_path
        except Exception as e:
            raise Exception(f"Failed to save variable '{var_name}': {e}")
    
    def save_hyperparameters(self, save_filename="hyperparameters.json"):
        """
        保存所有首字母大写的全局变量作为超参数
        
        Args:
            save_filename: 保存的文件名，默认为"hyperparameters.json"
        """
        hyperparameters = {}
        
        frame = inspect.currentframe()
        if frame and frame.f_back:
            caller_globals = frame.f_back.f_globals
        else:
            caller_globals = globals()
        
        for name, value in caller_globals.items():
            if not name[0].isupper() or name.startswith('_'):
                continue
            
            if name in ['In', 'Out']:
                continue
                
            if isinstance(value, types.ModuleType):
                continue
            
            if inspect.ismodule(value) or inspect.isclass(value) or inspect.isfunction(value) or inspect.ismethod(value):
                continue
            
            try:
                if isinstance(value, (int, float, str, bool, type(None))):
                    hyperparameters[name] = value
                elif isinstance(value, (list, tuple)):
                    if all(isinstance(item, (int, float, str, bool, type(None))) for item in value):
                        hyperparameters[name] = value
                    else:
                        hyperparameters[name] = str(value)
                elif isinstance(value, dict):
                    if all(isinstance(k, (int, float, str, bool)) and isinstance(v, (int, float, str, bool, type(None))) 
                           for k, v in value.items()):
                        hyperparameters[name] = value
                    else:
                        hyperparameters[name] = str(value)
                else:
                    hyperparameters[name] = str(value)
            except:
                continue
        
        save_path = self.log_dir / save_filename
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(hyperparameters, f, indent=4, ensure_ascii=False)
        
        print(f"\nHyperparameters saved to: {save_path}")
        print(f"Total {len(hyperparameters)} hyperparameters saved\n")
        
        if hyperparameters:
            print("-" * 80)
            print(f"{'Variable':<20} {'Type':<15} {'Value':<45}")
            print("-" * 80)
            
            for name, value in sorted(hyperparameters.items()):
                type_name = type(value).__name__
                
                if isinstance(value, str):
                    value_str = f'"{value}"' if len(value) <= 40 else f'"{value[:37]}..."'
                elif isinstance(value, (list, dict, tuple)):
                    value_str = str(value) if len(str(value)) <= 43 else str(value)[:40] + "..."
                else:
                    value_str = str(value)
                
                print(f"{name:<20} {type_name:<15} {value_str:<45}")
            
            print("-" * 80)
        
        return hyperparameters
    
    def rm_logs(self):
        """
        删除base_path下所有符合时间戳-项目名格式的文件夹
        """
        import re
        
        pattern = re.compile(r'^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\w+$')
        
        dirs_to_delete = []
        for item in self.base_path.iterdir():
            if item.is_dir() and pattern.match(item.name):
                dirs_to_delete.append(item)
        
        if not dirs_to_delete:
            print("No log directories found")
            return
        
        print(f"Found {len(dirs_to_delete)} log directories:")
        for dir_path in dirs_to_delete:
            print(f"  - {dir_path.name}")
        
        confirm = input("\nAre you sure to delete all these directories? (yes/no): ")
        
        if confirm.lower() == 'yes':
            deleted_count = 0
            for dir_path in dirs_to_delete:
                try:
                    shutil.rmtree(dir_path)
                    print(f"Deleted: {dir_path.name}")
                    deleted_count += 1
                except Exception as e:
                    print(f"Failed to delete {dir_path.name}: {e}")
            print(f"\nTotal deleted: {deleted_count} directories")
        else:
            print("Deletion cancelled")
