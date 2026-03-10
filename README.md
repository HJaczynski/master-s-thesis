# Computer Vision Model Benchmarking Tool 🚀

A comprehensive benchmarking suite for Computer Vision models that enables testing and comparing various optimization methods including quantization, pruning, and knowledge distillation.

## Features ✨

- **Multi-Model Support**: Benchmark popular CV architectures (ResNet, MobileNet, EfficientNet, VGG, DenseNet)
- **Optimization Methods**:
  - Quantization (INT8, FP16)
  - Pruning (L1 unstructured, random, structured)
  - Knowledge Distillation
  - Iterative pruning with fine-tuning
- **Comprehensive Metrics**:
  - Inference time and throughput (FPS)
  - Memory usage (peak and average)
  - Model size
  - Accuracy (optional with dataset)
- **Rich Visualizations**: Automatic generation of comparison plots and analysis charts
- **Flexible Configuration**: YAML-based configuration for easy experimentation

## Installation 📦

### Using uv (recommended)

```bash
# Clone the repository
git clone <your-repo-url>
cd master-s-thesis

# Install dependencies
uv sync
```

### Using pip

```bash
pip install torch torchvision pandas seaborn matplotlib numpy psutil pyyaml
```

## Quick Start 🏃

### Basic Usage

Run a quick benchmark on two models:

```bash
python main.py
# Select option 1 for quick benchmark
```

### Benchmark Baseline Models

```python
from src.benchmarks import BenchmarkEngine

# Initialize benchmark engine
engine = BenchmarkEngine(device='cuda')

# Benchmark multiple models
results = engine.benchmark_multiple_models(
    model_names=['resnet18', 'mobilenet_v2', 'efficientnet_b0'],
    num_iterations=100
)

# Print and save results
engine.print_summary_table()
engine.save_results('results.json')
```

### Compare Optimizations

```python
from src.models import ModelLoader
from src.optimizations import QuantizationOptimizer, PruningOptimizer

loader = ModelLoader(device='cuda')
model = loader.load_model('mobilenet_v2', pretrained=True)

# Apply optimizations
optimizations = {
    'baseline': model,
    'quantized': QuantizationOptimizer(dtype='int8').optimize(model),
    'pruned_30': PruningOptimizer(amount=0.3).optimize(model),
}

# Benchmark all versions
results = engine.compare_optimizations(
    model_name='mobilenet_v2',
    optimizations=optimizations
)
```

### Generate Visualizations

```python
from src.visualization import ResultVisualizer

visualizer = ResultVisualizer(output_dir='plots')
df = engine.get_results_dataframe()
visualizer.create_comprehensive_report(df)
```

## Project Structure 📁

```
master-s-thesis/
├── src/
│   ├── models/              # Model loading and management
│   │   └── model_loader.py
│   ├── benchmarks/          # Benchmarking engine and metrics
│   │   ├── benchmark_engine.py
│   │   └── metrics.py
│   ├── optimizations/       # Optimization methods
│   │   ├── base.py
│   │   ├── quantization.py
│   │   ├── pruning.py
│   │   └── distillation.py
│   └── visualization/       # Results visualization
│       └── visualizer.py
├── config/                  # Configuration files
│   ├── benchmark_config.py
│   └── example_config.yaml
├── results/                 # Benchmark results (JSON)
├── plots/                   # Generated visualizations
├── main.py                  # Main entry point
├── pyproject.toml          # Project dependencies
└── README.md               # This file
```

## Optimization Methods 🔧

### 1. Quantization

Reduce model precision for faster inference and smaller size:

```python
from src.optimizations import QuantizationOptimizer

# INT8 quantization
quantizer = QuantizationOptimizer(dtype='int8', backend='fbgemm')
quantized_model = quantizer.optimize(model)

# FP16 quantization
quantizer_fp16 = QuantizationOptimizer(dtype='fp16')
fp16_model = quantizer_fp16.optimize(model)
```

### 2. Pruning

Remove unnecessary weights to reduce model size:

```python
from src.optimizations import PruningOptimizer

# L1 unstructured pruning (30%)
pruner = PruningOptimizer(
    pruning_method='l1_unstructured',
    amount=0.3
)
pruned_model = pruner.optimize(model)

# Check sparsity
sparsity = pruner.get_sparsity(pruned_model)
print(f"Sparsity: {sparsity['sparsity_percentage']:.2f}%")
```

### 3. Knowledge Distillation

Train smaller models using knowledge from larger ones:

```python
from src.optimizations import KnowledgeDistillationOptimizer

distiller = KnowledgeDistillationOptimizer(temperature=3.0, alpha=0.5)
student_model = distiller.optimize(
    student_model=small_model,
    teacher_model=large_model,
    train_loader=data_loader,
    num_epochs=10
)
```

## Metrics Collected 📊

For each model and optimization, the tool collects:

- **Timing Metrics**:
  - Average inference time (ms)
  - Standard deviation
  - Throughput (FPS)
  
- **Memory Metrics**:
  - Peak memory usage (MB)
  - Average memory usage (MB)
  - Model size on disk (MB)
  
- **Quality Metrics** (optional):
  - Top-1 accuracy
  - Top-5 accuracy

## Visualization Examples 📈

The tool automatically generates:

1. **Inference Time Comparison**: Bar charts comparing inference times
2. **Throughput Comparison**: FPS comparison across models
3. **Memory Usage**: Peak memory and model size analysis
4. **Accuracy vs Speed**: Trade-off visualization
5. **Optimization Heatmaps**: Multi-metric comparison matrix
6. **Speedup Comparison**: Relative performance improvements

## Configuration 🔧

Use YAML configuration files for reproducible experiments:

```yaml
# config/example_config.yaml
device: 'cuda'

models:
  - 'resnet18'
  - 'mobilenet_v2'
  - 'efficientnet_b0'

optimizations:
  baseline:
    enabled: true
  
  quantization:
    enabled: true
    dtype: 'int8'
    backend: 'fbgemm'
  
  pruning:
    enabled: true
    method: 'l1_unstructured'
    amount: 0.3

num_iterations: 100
batch_size: 1
```

Load and use:

```python
from config.benchmark_config import BenchmarkConfig

config = BenchmarkConfig('config/example_config.yaml')
device = config['device']
models = config['models']
```

## Advanced Usage 🎓

### Custom Models

Benchmark your own models:

```python
from src.models import ModelLoader

loader = ModelLoader(device='cuda')
custom_model = YourCustomModel()
model = loader.load_custom_model(custom_model)

# Benchmark it
metrics = engine.benchmark_model(
    model=model,
    model_name='custom_model',
    input_shape=(1, 3, 224, 224)
)
```

### Accuracy Evaluation

Include accuracy metrics with a dataset:

```python
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Create data loader
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder('path/to/val', transform=transform)
data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

# Benchmark with accuracy
metrics = engine.benchmark_model(
    model=model,
    model_name='resnet18',
    data_loader=data_loader
)
```

### Iterative Pruning

Progressive pruning with fine-tuning:

```python
from src.optimizations import IterativePruningOptimizer

iterative_pruner = IterativePruningOptimizer(
    target_sparsity=0.5,
    num_iterations=5
)

# With fine-tuning function
def fine_tune(model):
    # Your training loop here
    return model

pruned_model = iterative_pruner.optimize(
    model=model,
    train_fn=fine_tune
)
```

## Requirements 📋

- Python >= 3.8
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- pandas >= 1.5.0
- seaborn >= 0.12.0
- matplotlib >= 3.5.0
- numpy >= 1.21.0
- psutil >= 5.9.0
- pyyaml >= 6.0

## Contributing 🤝

Contributions are welcome! Feel free to:

- Add new optimization methods
- Support additional model architectures
- Improve visualization capabilities
- Add new metrics
- Fix bugs and improve documentation

## Roadmap 🗺️

- [ ] Support for ONNX export and benchmarking
- [ ] TensorRT optimization integration
- [ ] Automated hyperparameter search for optimizations
- [ ] Multi-GPU benchmarking
- [ ] Web interface for result visualization
- [ ] Integration with MLflow for experiment tracking
- [ ] Support for object detection and segmentation models

## License 📄

MIT License - see LICENSE file for details.

## Citation 📚

If you use this tool in your research, please cite:

```bibtex
@software{cv_model_benchmark,
  title = {Computer Vision Model Benchmarking Tool},
  author = {Your Name},
  year = {2026},
  url = {https://github.com/yourusername/master-s-thesis}
}
```

## Acknowledgments 🙏

- PyTorch and torchvision teams for excellent deep learning frameworks
- The open-source community for various optimization techniques
- Research papers on model compression and optimization

## Contact 📧

For questions, issues, or suggestions:
- Open an issue on GitHub
- Email: your.email@example.com

---

**Happy Benchmarking! 🎯**