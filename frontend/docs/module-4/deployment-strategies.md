---
sidebar_position: 14
title: "Deployment Strategies"
---

# Deployment Strategies for VLA Systems

## Introduction to VLA Deployment

Deploying Vision-Language-Action (VLA) systems in humanoid robotics requires careful consideration of computational constraints, real-time performance requirements, safety considerations, and operational reliability. Unlike traditional AI systems, VLA systems must operate in real-time with safety-critical constraints while processing multiple modalities simultaneously. This module explores comprehensive deployment strategies for implementing VLA systems on humanoid robots.

## Hardware Deployment Considerations

### 1. Edge Computing Platforms for Humanoid Robots

Humanoid robots require powerful yet compact computing platforms that can handle the computational demands of VLA systems:

```python
# Hardware compatibility and deployment manager
import torch
import torch.nn as nn
import subprocess
import psutil
import GPUtil
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class HardwareSpecs:
    """Hardware specifications for VLA deployment"""
    platform_name: str
    cpu_cores: int
    cpu_frequency_ghz: float
    gpu_compute_capability: str
    gpu_memory_gb: float
    gpu_performance_tflops: float
    memory_gb: float
    storage_gb: float
    power_consumption_w: float
    thermal_design_power_w: float
    max_operating_temp_c: float
    supported_precision: List[str]  # ['fp32', 'fp16', 'int8', 'int4']

# Supported hardware platforms
HARDWARE_PLATFORMS = {
    'nvidia_jetson_orin': HardwareSpecs(
        platform_name='NVIDIA Jetson Orin AGX',
        cpu_cores=12,
        cpu_frequency_ghz=2.2,
        gpu_compute_capability='8.7',
        gpu_memory_gb=64,
        gpu_performance_tflops=275.0,
        memory_gb=32,
        storage_gb=128,
        power_consumption_w=60,
        thermal_design_power_w=60,
        max_operating_temp_c=85,
        supported_precision=['fp32', 'fp16', 'int8']
    ),
    'nvidia_jetson_xavier': HardwareSpecs(
        platform_name='NVIDIA Jetson Xavier NX',
        cpu_cores=8,
        cpu_frequency_ghz=1.9,
        gpu_compute_capability='7.2',
        gpu_memory_gb=8,
        gpu_performance_tflops=21.0,
        memory_gb=16,
        storage_gb=64,
        power_consumption_w=15,
        thermal_design_power_w=15,
        max_operating_temp_c=85,
        supported_precision=['fp32', 'fp16', 'int8']
    ),
    'intel_up_squared': HardwareSpecs(
        platform_name='Intel UP Squared',
        cpu_cores=4,
        cpu_frequency_ghz=1.5,
        gpu_compute_capability='cpu_only',
        gpu_memory_gb=0,
        gpu_performance_tflops=0.5,
        memory_gb=8,
        storage_gb=64,
        power_consumption_w=10,
        thermal_design_power_w=15,
        max_operating_temp_c=70,
        supported_precision=['fp32', 'fp16']
    ),
    'raspberry_pi_4': HardwareSpecs(
        platform_name='Raspberry Pi 4',
        cpu_cores=4,
        cpu_frequency_ghz=1.5,
        gpu_compute_capability='cpu_only',
        gpu_memory_gb=0,
        gpu_performance_tflops=0.1,
        memory_gb=4,
        storage_gb=32,
        power_consumption_w=6,
        thermal_design_power_w=6,
        max_operating_temp_c=85,
        supported_precision=['fp32']
    )
}

class HardwareCompatibilityManager:
    """Manage hardware compatibility for VLA deployments"""
    def __init__(self):
        self.available_hardware = HARDWARE_PLATFORMS
        self.current_hardware = self.detect_current_hardware()

    def detect_current_hardware(self) -> Optional[HardwareSpecs]:
        """Detect current hardware specifications"""
        try:
            # Get CPU information
            cpu_info = self.get_cpu_info()

            # Get GPU information
            gpu_info = self.get_gpu_info()

            # Get memory information
            memory_info = self.get_memory_info()

            # Get storage information
            storage_info = self.get_storage_info()

            # Determine compatible platform
            for platform_name, specs in self.available_hardware.items():
                if self.matches_hardware(specs, cpu_info, gpu_info, memory_info):
                    return specs

            # If no exact match, return best approximation
            return self.find_closest_hardware(cpu_info, gpu_info, memory_info)

        except Exception as e:
            print(f"Hardware detection failed: {e}")
            return None

    def get_cpu_info(self) -> Dict[str, any]:
        """Get CPU information"""
        return {
            'cores': psutil.cpu_count(logical=False),
            'threads': psutil.cpu_count(logical=True),
            'frequency': psutil.cpu_freq().max / 1000 if psutil.cpu_freq() else 2.0,
            'usage': psutil.cpu_percent(interval=1)
        }

    def get_gpu_info(self) -> Dict[str, any]:
        """Get GPU information"""
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]  # Use primary GPU
            return {
                'count': len(gpus),
                'memory': gpu.memoryTotal / 1024,  # Convert MB to GB
                'utilization': gpu.load,
                'compute_capability': self.get_compute_capability(gpu.name)
            }
        else:
            return {
                'count': 0,
                'memory': 0,
                'utilization': 0,
                'compute_capability': 'cpu_only'
            }

    def get_memory_info(self) -> Dict[str, any]:
        """Get memory information"""
        memory = psutil.virtual_memory()
        return {
            'total': memory.total / (1024**3),  # Convert to GB
            'available': memory.available / (1024**3),
            'used_percent': memory.percent
        }

    def get_storage_info(self) -> Dict[str, any]:
        """Get storage information"""
        disk = psutil.disk_usage('/')
        return {
            'total': disk.total / (1024**3),  # Convert to GB
            'free': disk.free / (1024**3),
            'used_percent': (disk.used / disk.total) * 100
        }

    def matches_hardware(self, expected: HardwareSpecs,
                         actual_cpu: Dict[str, any],
                         actual_gpu: Dict[str, any],
                         actual_memory: Dict[str, any]) -> bool:
        """Check if actual hardware matches expected specifications"""
        # Check CPU cores
        if actual_cpu['cores'] < expected.cpu_cores:
            return False

        # Check CPU frequency
        if actual_cpu['frequency'] < expected.cpu_frequency_ghz * 0.8:  # Allow 20% tolerance
            return False

        # Check GPU memory (if required)
        if expected.gpu_memory_gb > 0 and actual_gpu['memory'] < expected.gpu_memory_gb * 0.9:  # Allow 10% tolerance
            return False

        # Check memory
        if actual_memory['total'] < expected.memory_gb * 0.9:  # Allow 10% tolerance
            return False

        return True

    def find_closest_hardware(self, cpu_info: Dict[str, any],
                             gpu_info: Dict[str, any],
                             memory_info: Dict[str, any]) -> HardwareSpecs:
        """Find closest matching hardware platform"""
        best_match = None
        best_score = 0

        for platform_name, specs in self.available_hardware.items():
            score = self.calculate_hardware_compatibility_score(
                specs, cpu_info, gpu_info, memory_info
            )

            if score > best_score:
                best_score = score
                best_match = specs

        return best_match

    def calculate_hardware_compatibility_score(self, specs: HardwareSpecs,
                                             cpu_info: Dict[str, any],
                                             gpu_info: Dict[str, any],
                                             memory_info: Dict[str, any]) -> float:
        """Calculate compatibility score (0.0-1.0)"""
        score = 0.0

        # CPU compatibility (40% weight)
        cpu_score = min(1.0, cpu_info['cores'] / specs.cpu_cores)
        cpu_freq_score = min(1.0, cpu_info['frequency'] / specs.cpu_frequency_ghz)
        score += 0.4 * (0.7 * cpu_score + 0.3 * cpu_freq_score)

        # GPU compatibility (40% weight)
        if specs.gpu_memory_gb > 0 and gpu_info['memory'] > 0:
            gpu_memory_score = min(1.0, gpu_info['memory'] / specs.gpu_memory_gb)
            gpu_compute_score = 1.0 if gpu_info['compute_capability'] >= specs.gpu_compute_capability else 0.5
            score += 0.4 * (0.6 * gpu_memory_score + 0.4 * gpu_compute_score)
        elif specs.gpu_memory_gb == 0:  # CPU-only platform
            score += 0.4  # Full credit for CPU-only requirement

        # Memory compatibility (20% weight)
        memory_score = min(1.0, memory_info['total'] / specs.memory_gb)
        score += 0.2 * memory_score

        return score

    def recommend_deployment_platform(self, vla_model_requirements: Dict[str, any]) -> str:
        """Recommend best deployment platform for VLA model"""
        best_platform = None
        best_compatibility = 0.0

        for platform_name, specs in self.available_hardware.items():
            compatibility = self.evaluate_model_compatibility(
                specs, vla_model_requirements
            )

            if compatibility > best_compatibility:
                best_compatibility = compatibility
                best_platform = platform_name

        return best_platform, best_compatibility

    def evaluate_model_compatibility(self, hardware: HardwareSpecs,
                                   model_requirements: Dict[str, any]) -> float:
        """Evaluate if hardware can support model requirements"""
        score = 0.0
        weight_sum = 0.0

        # GPU memory requirement (50% weight)
        if 'gpu_memory_requirement_gb' in model_requirements:
            req_memory = model_requirements['gpu_memory_requirement_gb']
            if hardware.gpu_memory_gb >= req_memory:
                gpu_memory_score = 1.0
            else:
                gpu_memory_score = hardware.gpu_memory_gb / req_memory
            score += 0.5 * gpu_memory_score
            weight_sum += 0.5

        # Compute requirement (30% weight)
        if 'compute_requirement_tflops' in model_requirements:
            req_compute = model_requirements['compute_requirement_tflops']
            if hardware.gpu_performance_tflops >= req_compute:
                compute_score = 1.0
            else:
                compute_score = hardware.gpu_performance_tflops / req_compute
            score += 0.3 * compute_score
            weight_sum += 0.3

        # Memory requirement (20% weight)
        if 'memory_requirement_gb' in model_requirements:
            req_memory = model_requirements['memory_requirement_gb']
            if hardware.memory_gb >= req_memory:
                memory_score = 1.0
            else:
                memory_score = hardware.memory_gb / req_memory
            score += 0.2 * memory_score
            weight_sum += 0.2

        return score / weight_sum if weight_sum > 0 else 0.0

class ModelOptimizer:
    """Optimize VLA models for specific hardware platforms"""
    def __init__(self, hardware_specs: HardwareSpecs):
        self.hardware_specs = hardware_specs
        self.optimization_strategies = self.get_optimization_strategies()

    def get_optimization_strategies(self) -> Dict[str, any]:
        """Get optimization strategies based on hardware"""
        strategies = {
            'quantization': self.supports_quantization(),
            'pruning': self.supports_pruning(),
            'tensor_parallelism': self.supports_tensor_parallelism(),
            'model_partitioning': self.supports_model_partitioning(),
            'precision_optimization': self.get_precision_optimization()
        }
        return strategies

    def supports_quantization(self) -> bool:
        """Check if hardware supports quantization"""
        return 'int8' in self.hardware_specs.supported_precision

    def supports_pruning(self) -> bool:
        """Check if hardware supports pruning"""
        return True  # Most hardware supports pruning

    def supports_tensor_parallelism(self) -> bool:
        """Check if hardware supports tensor parallelism"""
        return self.hardware_specs.gpu_compute_capability != 'cpu_only'

    def supports_model_partitioning(self) -> bool:
        """Check if hardware supports model partitioning"""
        return self.hardware_specs.memory_gb >= 8  # Need at least 8GB for partitioning

    def get_precision_optimization(self) -> str:
        """Get recommended precision optimization"""
        if self.hardware_specs.gpu_compute_capability == 'cpu_only':
            return 'fp32'  # CPU typically performs better with FP32
        elif self.hardware_specs.gpu_compute_capability >= '7.0':
            return 'fp16'  # Modern GPUs support FP16 well
        else:
            return 'int8'  # Fallback to INT8 for older hardware

    def optimize_model_for_hardware(self, model: nn.Module) -> nn.Module:
        """Optimize model for specific hardware"""
        optimized_model = model

        # Apply precision optimization
        if self.optimization_strategies['precision_optimization'] == 'fp16':
            optimized_model = self.apply_fp16_optimization(optimized_model)
        elif self.optimization_strategies['precision_optimization'] == 'int8':
            optimized_model = self.apply_int8_optimization(optimized_model)

        # Apply quantization if supported
        if self.optimization_strategies['quantization']:
            optimized_model = self.apply_quantization(optimized_model)

        # Apply pruning if supported
        if self.optimization_strategies['pruning']:
            optimized_model = self.apply_pruning(optimized_model)

        # Apply tensor parallelism if supported
        if self.optimization_strategies['tensor_parallelism']:
            optimized_model = self.apply_tensor_parallelism(optimized_model)

        # Apply model partitioning if supported
        if self.optimization_strategies['model_partitioning']:
            optimized_model = self.apply_model_partitioning(optimized_model)

        return optimized_model

    def apply_fp16_optimization(self, model: nn.Module) -> nn.Module:
        """Apply FP16 optimization"""
        # Convert model to FP16 where appropriate
        model = model.half()

        # Ensure certain layers remain in FP32 (like batch norm)
        for name, module in model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module = module.float()  # Keep batch norm in FP32

        return model

    def apply_int8_optimization(self, model: nn.Module) -> nn.Module:
        """Apply INT8 optimization"""
        # Use PyTorch's quantization tools
        model.eval()

        # Specify quantization configuration
        model.qconfig = torch.quantization.get_default_qconfig('qnnpack')

        # Prepare model for quantization
        torch.quantization.prepare(model, inplace=True)

        # Convert to quantized model
        torch.quantization.convert(model, inplace=True)

        return model

    def apply_quantization(self, model: nn.Module) -> nn.Module:
        """Apply quantization to model"""
        if 'int8' in self.hardware_specs.supported_precision:
            return self.apply_int8_optimization(model)
        elif 'int4' in self.hardware_specs.supported_precision:
            return self.apply_int4_optimization(model)
        else:
            return model

    def apply_pruning(self, model: nn.Module) -> nn.Module:
        """Apply pruning to model"""
        import torch.nn.utils.prune as prune

        # Apply unstructured pruning to linear layers
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=0.2)  # Prune 20%

        return model

    def apply_tensor_parallelism(self, model: nn.Module) -> nn.Module:
        """Apply tensor parallelism to model"""
        if torch.cuda.device_count() > 1:
            # Use DataParallel for multi-GPU systems
            model = nn.DataParallel(model)

        return model

    def apply_model_partitioning(self, model: nn.Module) -> nn.Module:
        """Apply model partitioning for memory-constrained systems"""
        # This would involve more complex partitioning logic
        # For now, return the model as-is
        return model

    def estimate_inference_performance(self, model: nn.Module) -> Dict[str, float]:
        """Estimate inference performance on hardware"""
        # Create dummy inputs matching expected sizes
        dummy_vision_input = torch.randn(1, 3, 224, 224).to('cuda' if torch.cuda.is_available() else 'cpu')
        dummy_language_input = torch.randint(0, 1000, (1, 32)).to('cuda' if torch.cuda.is_available() else 'cpu')

        # Warm up
        model.eval()
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_vision_input, dummy_language_input)

        # Measure inference time
        import time
        start_time = time.time()
        with torch.no_grad():
            for _ in range(100):  # 100 inferences
                _ = model(dummy_vision_input, dummy_language_input)
        end_time = time.time()

        avg_inference_time = (end_time - start_time) / 100
        fps = 1.0 / avg_inference_time if avg_inference_time > 0 else float('inf')

        return {
            'avg_inference_time_ms': avg_inference_time * 1000,
            'fps': fps,
            'estimated_memory_usage_mb': self.estimate_memory_usage(model),
            'power_estimation_w': self.estimate_power_consumption(model)
        }

    def estimate_memory_usage(self, model: nn.Module) -> float:
        """Estimate memory usage of model"""
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        total_size = param_size + buffer_size
        return total_size / (1024 * 1024)  # Convert to MB

    def estimate_power_consumption(self, model: nn.Module) -> float:
        """Estimate power consumption based on model size and hardware"""
        memory_usage_mb = self.estimate_memory_usage(model)
        base_power = self.hardware_specs.power_consumption_w

        # Power scales with memory usage and compute intensity
        power_multiplier = 1.0 + (memory_usage_mb / 1000.0) * 0.1  # 10% increase per GB

        return base_power * power_multiplier
```

### 2. Real-time Performance Optimization

```python
# Real-time performance optimization for VLA systems
import threading
import queue
import time
from collections import deque
import multiprocessing as mp

class RealTimePerformanceOptimizer:
    """Optimize VLA systems for real-time performance"""
    def __init__(self, target_fps=30, max_latency_ms=50):
        self.target_fps = target_fps
        self.max_latency_ms = max_latency_ms
        self.performance_monitor = PerformanceMonitor()
        self.resource_manager = ResourceManager()
        self.dynamic_batching = DynamicBatching()
        self.pipeline_scheduler = PipelineScheduler()

    def optimize_for_realtime(self, vla_model: nn.Module) -> nn.Module:
        """Apply real-time optimizations to VLA model"""
        # Apply tensor optimization
        optimized_model = self.optimize_tensors(vla_model)

        # Apply memory optimization
        optimized_model = self.optimize_memory(optimized_model)

        # Apply computational optimization
        optimized_model = self.optimize_computation(optimized_model)

        # Apply pipeline optimization
        optimized_model = self.optimize_pipeline(optimized_model)

        return optimized_model

    def optimize_tensors(self, model: nn.Module) -> nn.Module:
        """Optimize tensor operations"""
        # Use TensorRT for NVIDIA GPUs
        if torch.cuda.is_available() and self.is_nvidia_gpu():
            model = self.apply_tensorrt_optimization(model)
        else:
            # Use ONNX Runtime for other systems
            model = self.apply_onnx_optimization(model)

        return model

    def apply_tensorrt_optimization(self, model: nn.Module) -> nn.Module:
        """Apply TensorRT optimization"""
        try:
            import tensorrt as trt
            from torch2trt import torch2trt

            # Create dummy inputs for optimization
            dummy_vision = torch.randn(1, 3, 224, 224).cuda()
            dummy_language = torch.randint(0, 1000, (1, 32)).cuda()

            # Optimize with TensorRT
            optimized_model = torch2trt(
                model,
                [dummy_vision, dummy_language],
                fp16_mode=True,
                max_workspace_size=1<<30  # 1GB
            )

            return optimized_model
        except ImportError:
            print("TensorRT not available, using standard optimization")
            return model

    def apply_onnx_optimization(self, model: nn.Module) -> nn.Module:
        """Apply ONNX optimization"""
        try:
            import onnxruntime as ort
            import onnx

            # Export to ONNX
            dummy_vision = torch.randn(1, 3, 224, 224)
            dummy_language = torch.randint(0, 1000, (1, 32))

            torch.onnx.export(
                model,
                (dummy_vision, dummy_language),
                "temp_vla_model.onnx",
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['vision_input', 'language_input'],
                output_names=['action_output']
            )

            # Load with ONNX Runtime
            ort_session = ort.InferenceSession("temp_vla_model.onnx")

            # Create wrapper for ONNX model
            class ONNXModelWrapper(nn.Module):
                def __init__(self, session):
                    super().__init__()
                    self.session = session

                def forward(self, vision_input, language_input):
                    # Run inference with ONNX Runtime
                    inputs = {
                        'vision_input': vision_input.cpu().numpy(),
                        'language_input': language_input.cpu().numpy()
                    }
                    outputs = self.session.run(None, inputs)
                    return torch.from_numpy(outputs[0]).to(vision_input.device)

            return ONNXModelWrapper(ort_session)

        except ImportError:
            print("ONNX Runtime not available")
            return model

    def optimize_memory(self, model: nn.Module) -> nn.Module:
        """Optimize memory usage"""
        # Enable gradient checkpointing for memory efficiency
        self.apply_gradient_checkpointing(model)

        # Optimize tensor memory layout
        self.optimize_tensor_layout(model)

        return model

    def apply_gradient_checkpointing(self, model: nn.Module):
        """Apply gradient checkpointing to save memory"""
        # This would apply gradient checkpointing to appropriate layers
        # For VLA models, we might checkpoint the transformer layers
        for name, module in model.named_modules():
            if 'transformer' in name.lower() or 'attention' in name.lower():
                # Apply checkpointing to transformer blocks
                pass  # Implementation would depend on model architecture

    def optimize_tensor_layout(self, model: nn.Module):
        """Optimize tensor memory layout"""
        # Convert to channels_last format for better memory access
        model = model.to(memory_format=torch.channels_last)

        return model

    def optimize_computation(self, model: nn.Module) -> nn.Module:
        """Optimize computation"""
        # Enable fused operations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

        # Optimize for inference
        model.eval()
        model = torch.jit.optimize_for_inference(torch.jit.script(model))

        return model

    def optimize_pipeline(self, model: nn.Module) -> nn.Module:
        """Optimize pipeline execution"""
        # Create pipeline-parallel model
        pipeline_model = PipelineModel(model)

        return pipeline_model

    def is_nvidia_gpu(self) -> bool:
        """Check if system has NVIDIA GPU"""
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0).lower()
            return 'nvidia' in gpu_name
        return False

class PerformanceMonitor:
    """Monitor real-time performance"""
    def __init__(self):
        self.inference_times = deque(maxlen=1000)
        self.memory_usage = deque(maxlen=1000)
        self.cpu_usage = deque(maxlen=1000)
        self.gpu_usage = deque(maxlen=1000)
        self.latency_history = deque(maxlen=1000)
        self.throughput_history = deque(maxlen=1000)

    def record_inference_time(self, time_ms: float):
        """Record inference time"""
        self.inference_times.append(time_ms)

    def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics"""
        if not self.inference_times:
            return {
                'avg_inference_time_ms': float('inf'),
                'current_fps': 0.0,
                'latency_percentile_95': float('inf'),
                'memory_usage_avg': 0.0,
                'cpu_usage_avg': 0.0,
                'gpu_usage_avg': 0.0
            }

        avg_inference_time = sum(self.inference_times) / len(self.inference_times)
        current_fps = 1000.0 / avg_inference_time if avg_inference_time > 0 else 0.0

        # Calculate 95th percentile latency
        sorted_times = sorted(list(self.inference_times))
        p95_idx = int(0.95 * len(sorted_times))
        p95_latency = sorted_times[min(p95_idx, len(sorted_times) - 1)] if sorted_times else float('inf')

        return {
            'avg_inference_time_ms': avg_inference_time,
            'current_fps': current_fps,
            'latency_percentile_95': p95_latency,
            'memory_usage_avg': sum(self.memory_usage) / len(self.memory_usage) if self.memory_usage else 0.0,
            'cpu_usage_avg': sum(self.cpu_usage) / len(self.cpu_usage) if self.cpu_usage else 0.0,
            'gpu_usage_avg': sum(self.gpu_usage) / len(self.gpu_usage) if self.gpu_usage else 0.0
        }

    def is_performance_degrading(self) -> bool:
        """Check if performance is degrading"""
        if len(self.inference_times) < 10:
            return False

        # Compare recent performance with historical performance
        recent_avg = sum(list(self.inference_times)[-10:]) / 10
        historical_avg = sum(list(self.inference_times)[:-10]) / max(1, len(self.inference_times) - 10)

        # Performance is degrading if recent is significantly worse
        return recent_avg > historical_avg * 1.2  # 20% degradation threshold

    def get_performance_alerts(self) -> List[str]:
        """Get performance alerts"""
        alerts = []

        metrics = self.get_performance_metrics()

        if metrics['avg_inference_time_ms'] > 50:  # 50ms threshold
            alerts.append(f"High inference time: {metrics['avg_inference_time_ms']:.2f}ms")

        if metrics['current_fps'] < 20:  # 20 FPS threshold
            alerts.append(f"Low frame rate: {metrics['current_fps']:.2f} FPS")

        if metrics['latency_percentile_95'] > 100:  # 100ms threshold
            alerts.append(f"High latency (95th percentile): {metrics['latency_percentile_95']:.2f}ms")

        if metrics['gpu_usage_avg'] > 90:  # 90% threshold
            alerts.append(f"High GPU utilization: {metrics['gpu_usage_avg']:.2f}%")

        if metrics['memory_usage_avg'] > 85:  # 85% threshold
            alerts.append(f"High memory utilization: {metrics['memory_usage_avg']:.2f}%")

        return alerts

class ResourceManager:
    """Manage system resources for real-time execution"""
    def __init__(self):
        self.resource_allocation = {}
        self.priority_scheduler = PriorityScheduler()
        self.memory_allocator = MemoryAllocator()

    def allocate_resources(self, vla_model, hardware_specs: HardwareSpecs) -> Dict[str, any]:
        """Allocate resources for VLA model"""
        resource_plan = {}

        # Calculate memory requirements
        model_memory = self.estimate_model_memory(vla_model)
        available_memory = hardware_specs.memory_gb * 1024  # Convert to MB

        # Allocate GPU memory
        if hardware_specs.gpu_memory_gb > 0:
            gpu_memory_allocation = min(
                model_memory * 1.5,  # Include buffer for activations
                hardware_specs.gpu_memory_gb * 1024 * 0.8  # Use 80% of GPU memory
            )
            resource_plan['gpu_memory'] = gpu_memory_allocation

        # Allocate CPU resources
        cpu_allocation = self.calculate_cpu_allocation(hardware_specs)
        resource_plan['cpu_cores'] = cpu_allocation['cores']
        resource_plan['cpu_affinity'] = cpu_allocation['affinity']

        # Allocate I/O resources
        resource_plan['io_bandwidth'] = self.calculate_io_requirements()

        return resource_plan

    def estimate_model_memory(self, model: nn.Module) -> float:
        """Estimate model memory requirements in MB"""
        param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_memory = sum(b.numel() * b.element_size() for b in model.buffers())
        total_memory = param_memory + buffer_memory

        # Include activation memory (rough estimate: 3x parameter memory)
        activation_memory = total_memory * 3

        return (total_memory + activation_memory) / (1024 * 1024)  # Convert to MB

    def calculate_cpu_allocation(self, hardware_specs: HardwareSpecs) -> Dict[str, any]:
        """Calculate CPU allocation strategy"""
        # Reserve some cores for system processes
        reserved_cores = max(1, hardware_specs.cpu_cores // 4)
        available_cores = hardware_specs.cpu_cores - reserved_cores

        # Assign cores for different tasks
        allocation = {
            'vision_processing': max(1, available_cores // 3),
            'language_processing': max(1, available_cores // 3),
            'action_generation': max(1, available_cores // 3),
            'system_processes': reserved_cores
        }

        return {
            'cores': allocation,
            'affinity': self.create_cpu_affinity_mask(allocation)
        }

    def create_cpu_affinity_mask(self, allocation: Dict[str, int]) -> Dict[str, List[int]]:
        """Create CPU affinity masks for different processes"""
        core_assignment = {}
        core_offset = 0

        for process, core_count in allocation.items():
            core_assignment[process] = list(range(core_offset, core_offset + core_count))
            core_offset += core_count

        return core_assignment

    def calculate_io_requirements(self) -> float:
        """Calculate I/O bandwidth requirements"""
        # For VLA systems, I/O includes:
        # - Camera feeds (multiple cameras)
        # - Sensor data
        # - Command inputs
        # - Action outputs

        # Estimate based on typical humanoid robot I/O
        camera_bandwidth = 30 * 2 * 2  # 30 FPS, 2 cameras, 2 MB/frame (compressed)
        sensor_bandwidth = 100 * 0.1  # 100 Hz, 0.1 MB/s per sensor
        command_bandwidth = 10 * 0.01  # 10 Hz, 0.01 MB/s for commands

        total_bandwidth = camera_bandwidth + sensor_bandwidth + command_bandwidth

        return total_bandwidth

    def optimize_for_latency(self, model: nn.Module) -> nn.Module:
        """Optimize model for low latency"""
        # Apply optimizations that reduce latency
        model = self.apply_latency_optimizations(model)

        # Optimize for inference
        model.eval()

        # Apply JIT compilation
        model = torch.jit.optimize_for_inference(torch.jit.script(model))

        return model

    def apply_latency_optimizations(self, model: nn.Module) -> nn.Module:
        """Apply specific latency optimizations"""
        # Enable TensorRT optimizations if available
        if self.is_tensorrt_available():
            model = self.apply_tensorrt_latency_optimizations(model)

        # Optimize for memory access patterns
        model = self.optimize_memory_access(model)

        # Apply kernel fusion optimizations
        model = self.apply_kernel_fusion(model)

        return model

    def optimize_memory_access(self, model: nn.Module) -> nn.Module:
        """Optimize memory access patterns"""
        # Convert to channels_last format for better memory access on certain architectures
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
                module.to(memory_format=torch.channels_last)

        return model

    def apply_kernel_fusion(self, model: nn.Module) -> nn.Module:
        """Apply kernel fusion optimizations"""
        # This would apply various kernel fusion techniques
        # For PyTorch, we can use torch.jit.fuser
        torch.jit.enable_onednn_fusion(True)

        return model

    def is_tensorrt_available(self) -> bool:
        """Check if TensorRT is available"""
        try:
            import tensorrt
            return True
        except ImportError:
            return False

    def apply_tensorrt_latency_optimizations(self, model: nn.Module) -> nn.Module:
        """Apply TensorRT optimizations for latency"""
        try:
            import tensorrt as trt
            from torch2trt import torch2trt

            # Create dummy inputs
            dummy_vision = torch.randn(1, 3, 224, 224).cuda()
            dummy_language = torch.randint(0, 1000, (1, 32)).cuda()

            # Optimize with TensorRT for latency
            optimized_model = torch2trt(
                model,
                [dummy_vision, dummy_language],
                fp16_mode=True,
                int8_mode=True,
                max_workspace_size=1<<28,  # 256MB for low latency
                strict_type_constraints=True
            )

            return optimized_model
        except Exception as e:
            print(f"TensorRT optimization failed: {e}")
            return model

class DynamicBatching:
    """Dynamic batching for variable input sizes"""
    def __init__(self):
        self.batch_queue = queue.Queue()
        self.max_batch_size = 8
        self.batch_timeout = 0.01  # 10ms timeout
        self.batch_scheduler = BatchScheduler()

    def process_batch(self, inputs: List[Dict[str, any]]) -> List[any]:
        """Process batch of inputs with dynamic sizing"""
        if not inputs:
            return []

        # Group similar inputs together
        grouped_inputs = self.group_similar_inputs(inputs)

        results = []
        for group in grouped_inputs:
            # Process each group as a batch
            group_results = self.process_group_batch(group)
            results.extend(group_results)

        return results

    def group_similar_inputs(self, inputs: List[Dict[str, any]]) -> List[List[Dict[str, any]]]:
        """Group similar inputs for efficient batching"""
        # Group by input characteristics (size, complexity, etc.)
        groups = {}
        for input_data in inputs:
            # Create a signature based on input characteristics
            signature = self.create_input_signature(input_data)
            if signature not in groups:
                groups[signature] = []
            groups[signature].append(input_data)

        return list(groups.values())

    def create_input_signature(self, input_data: Dict[str, any]) -> str:
        """Create signature for input grouping"""
        # Create signature based on input characteristics
        vision_shape = input_data.get('vision_input', torch.tensor([])).shape
        language_length = input_data.get('language_input', torch.tensor([])).shape[1] if len(input_data.get('language_input', torch.tensor([])).shape) > 1 else 0

        return f"{vision_shape[1:]}_{language_length}"  # Exclude batch dimension

    def process_group_batch(self, group: List[Dict[str, any]]) -> List[any]:
        """Process a group of similar inputs as a batch"""
        if len(group) == 1:
            # Process single input
            return [self.process_single_input(group[0])]

        # Pad inputs to same size
        padded_inputs = self.pad_inputs_to_batch(group)

        # Process batch
        batch_results = self.process_batch_inputs(padded_inputs)

        # Unpad results
        unpadded_results = self.unpad_batch_results(batch_results, group)

        return unpadded_results

    def pad_inputs_to_batch(self, inputs: List[Dict[str, any]]) -> Dict[str, torch.Tensor]:
        """Pad inputs to create a batch"""
        # Find maximum dimensions
        max_vision_shape = [0, 0, 0, 0]  # [B, C, H, W]
        max_language_len = 0

        for input_data in inputs:
            vision_shape = input_data['vision_input'].shape
            language_len = input_data['language_input'].shape[1]

            max_vision_shape = [
                max(max_vision_shape[0], vision_shape[0]),
                max(max_vision_shape[1], vision_shape[1]),
                max(max_vision_shape[2], vision_shape[2]),
                max(max_vision_shape[3], vision_shape[3])
            ]
            max_language_len = max(max_language_len, language_len)

        # Create padded batch
        batch_size = len(inputs)
        padded_vision = torch.zeros(batch_size, *max_vision_shape[1:])
        padded_language = torch.zeros(batch_size, max_language_len, dtype=torch.long)

        for i, input_data in enumerate(inputs):
            # Copy vision input
            orig_vision = input_data['vision_input']
            padded_vision[i, :orig_vision.shape[0], :orig_vision.shape[1], :orig_vision.shape[2], :orig_vision.shape[3]] = orig_vision

            # Copy language input
            orig_language = input_data['language_input']
            padded_language[i, :orig_language.shape[1]] = orig_language

        return {
            'vision_input': padded_vision,
            'language_input': padded_language,
            'original_shapes': [inp['vision_input'].shape for inp in inputs]
        }

    def process_batch_inputs(self, batch_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Process batched inputs"""
        # This would call the actual model
        # For now, return mock results
        batch_size = batch_inputs['vision_input'].shape[0]
        action_dim = 14  # Example action dimension for humanoid
        return torch.randn(batch_size, action_dim)

    def unpad_batch_results(self, batch_results: torch.Tensor,
                           original_inputs: List[Dict[str, any]]) -> List[any]:
        """Unpad batch results to individual results"""
        results = []
        for i, orig_input in enumerate(original_inputs):
            # Extract result for this input
            result = batch_results[i]
            results.append(result)
        return results

    def process_single_input(self, input_data: Dict[str, any]) -> any:
        """Process single input"""
        # This would call the model on single input
        # For now, return mock result
        return torch.randn(14)  # Example action vector

class PipelineScheduler:
    """Schedule pipeline operations for optimal performance"""
    def __init__(self):
        self.pipeline_stages = []
        self.stage_scheduling = {}
        self.resource_allocation = {}

    def create_optimized_pipeline(self, model: nn.Module) -> nn.Module:
        """Create optimized pipeline from model"""
        # Analyze model structure
        model_analysis = self.analyze_model_structure(model)

        # Create pipeline stages based on analysis
        pipeline_stages = self.create_pipeline_stages(model_analysis)

        # Schedule stages for optimal execution
        scheduled_pipeline = self.schedule_pipeline_stages(pipeline_stages)

        return scheduled_pipeline

    def analyze_model_structure(self, model: nn.Module) -> Dict[str, any]:
        """Analyze model structure for pipelining"""
        analysis = {
            'modules': [],
            'connections': [],
            'compute_complexity': {},
            'memory_requirements': {},
            'execution_order': []
        }

        for name, module in model.named_modules():
            if not list(module.children()):  # Leaf modules only
                module_info = {
                    'name': name,
                    'type': type(module).__name__,
                    'parameters': sum(p.numel() for p in module.parameters()),
                    'compute_complexity': self.estimate_compute_complexity(module)
                }
                analysis['modules'].append(module_info)

        return analysis

    def estimate_compute_complexity(self, module: nn.Module) -> float:
        """Estimate compute complexity of module"""
        # This would estimate FLOPs, memory access, etc.
        # For now, return a simple estimate based on parameters
        param_count = sum(p.numel() for p in module.parameters())
        return param_count / 1000000.0  # Complexity in millions of parameters

    def create_pipeline_stages(self, model_analysis: Dict[str, any]) -> List[Dict[str, any]]:
        """Create pipeline stages from model analysis"""
        stages = []

        # Group modules by compute complexity and memory requirements
        modules = model_analysis['modules']
        modules_sorted = sorted(modules, key=lambda x: x['compute_complexity'], reverse=True)

        current_stage = []
        current_complexity = 0
        stage_threshold = 10.0  # Adjust based on target hardware

        for module in modules_sorted:
            if current_complexity + module['compute_complexity'] > stage_threshold:
                # Start new stage
                if current_stage:
                    stages.append({
                        'modules': current_stage,
                        'complexity': current_complexity,
                        'stage_id': len(stages)
                    })
                current_stage = [module]
                current_complexity = module['compute_complexity']
            else:
                current_stage.append(module)
                current_complexity += module['compute_complexity']

        # Add final stage
        if current_stage:
            stages.append({
                'modules': current_stage,
                'complexity': current_complexity,
                'stage_id': len(stages)
            })

        return stages

    def schedule_pipeline_stages(self, stages: List[Dict[str, any]]) -> nn.Module:
        """Schedule pipeline stages for execution"""
        # This would create a pipeline-scheduled model
        # For now, return the original model with pipeline annotations
        return PipelineModelWrapper(stages)

class PipelineModelWrapper(nn.Module):
    """Wrapper for pipeline-scheduled model"""
    def __init__(self, pipeline_stages: List[Dict[str, any]]):
        super().__init__()
        self.stages = nn.ModuleList()
        self.stage_schedulers = []

        for stage_info in pipeline_stages:
            stage_modules = nn.Sequential()
            for module_info in stage_info['modules']:
                # Add module to stage (this is simplified)
                pass
            self.stages.append(stage_modules)
            self.stage_schedulers.append(StageScheduler(stage_info))

    def forward(self, vision_input, language_input):
        """Forward pass through pipeline stages"""
        stage_outputs = []

        # Execute stages in pipeline fashion
        for i, stage in enumerate(self.stages):
            if i == 0:
                # First stage: combine inputs
                current_output = stage(vision_input, language_input)
            else:
                # Subsequent stages: process previous output
                current_output = stage(stage_outputs[-1])

            stage_outputs.append(current_output)

        return stage_outputs[-1]  # Return final output
```

## Deployment Architecture Patterns

### 1. Microservices Architecture for VLA Systems

```python
# Microservices architecture for VLA deployment
import asyncio
import aiohttp
from typing import Dict, List, Any, Optional
import json
import threading
import multiprocessing as mp

class VLAMicroservice:
    """Base class for VLA microservices"""
    def __init__(self, service_name: str, config: Dict[str, any]):
        self.service_name = service_name
        self.config = config
        self.running = False
        self.health_status = 'unknown'

    async def start_service(self):
        """Start the microservice"""
        self.running = True
        self.health_status = 'starting'
        await self.initialize()
        self.health_status = 'running'
        print(f"{self.service_name} started successfully")

    async def stop_service(self):
        """Stop the microservice"""
        self.running = False
        await self.cleanup()
        self.health_status = 'stopped'
        print(f"{self.service_name} stopped successfully")

    async def initialize(self):
        """Initialize service-specific resources"""
        pass

    async def cleanup(self):
        """Clean up service-specific resources"""
        pass

    async def process_request(self, request_data: Dict[str, any]) -> Dict[str, any]:
        """Process request - to be implemented by subclasses"""
        raise NotImplementedError

    def get_health_status(self) -> Dict[str, any]:
        """Get health status of the service"""
        return {
            'service': self.service_name,
            'status': self.health_status,
            'timestamp': time.time(),
            'config': self.config
        }

class VisionService(VLAMicroservice):
    """Vision processing microservice"""
    def __init__(self, config: Dict[str, any]):
        super().__init__('vision_service', config)
        self.vision_model = None
        self.feature_extractor = None

    async def initialize(self):
        """Initialize vision service"""
        # Load vision model
        model_path = self.config.get('vision_model_path', 'models/vision_model.pt')
        self.vision_model = torch.load(model_path)
        self.vision_model.eval()

        # Initialize feature extractor
        self.feature_extractor = self.create_feature_extractor()

    async def process_request(self, request_data: Dict[str, any]) -> Dict[str, any]:
        """Process vision request"""
        if 'image' not in request_data:
            return {'error': 'No image provided'}

        image_tensor = self.preprocess_image(request_data['image'])

        with torch.no_grad():
            features = self.vision_model(image_tensor)
            detections = self.extract_detections(features)

        return {
            'features': features.cpu().numpy().tolist(),
            'detections': detections,
            'timestamp': time.time()
        }

    def preprocess_image(self, image_data: any) -> torch.Tensor:
        """Preprocess image for vision model"""
        # Convert image data to tensor
        if isinstance(image_data, str):  # File path
            import cv2
            image = cv2.imread(image_data)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image_data, np.ndarray):
            image = image_data
        else:
            image = np.array(image_data)

        # Resize and normalize
        image = cv2.resize(image, (224, 224))
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))  # CHW format
        image = torch.from_numpy(image).unsqueeze(0)  # Add batch dimension

        return image.cuda() if torch.cuda.is_available() else image

    def extract_detections(self, features: torch.Tensor) -> List[Dict[str, any]]:
        """Extract object detections from features"""
        # This would implement actual detection extraction
        # For now, return mock detections
        return [
            {'class': 'person', 'confidence': 0.95, 'bbox': [100, 100, 200, 200]},
            {'class': 'chair', 'confidence': 0.87, 'bbox': [300, 200, 400, 300]}
        ]

class LanguageService(VLAMicroservice):
    """Language processing microservice"""
    def __init__(self, config: Dict[str, any]):
        super().__init__('language_service', config)
        self.language_model = None
        self.tokenizer = None

    async def initialize(self):
        """Initialize language service"""
        from transformers import AutoTokenizer, AutoModel

        model_name = self.config.get('language_model_name', 'bert-base-uncased')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.language_model = AutoModel.from_pretrained(model_name)
        self.language_model.eval()

    async def process_request(self, request_data: Dict[str, any]) -> Dict[str, any]:
        """Process language request"""
        if 'command' not in request_data:
            return {'error': 'No command provided'}

        command = request_data['command']

        # Tokenize command
        inputs = self.tokenizer(
            command,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=128
        )

        with torch.no_grad():
            outputs = self.language_model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)  # Average pooling

        # Parse command structure
        parsed_command = self.parse_command_structure(command)

        return {
            'embeddings': embeddings.cpu().numpy().tolist(),
            'parsed_command': parsed_command,
            'timestamp': time.time()
        }

    def parse_command_structure(self, command: str) -> Dict[str, any]:
        """Parse command structure into action-object-location format"""
        # Simple parsing - in practice, use NLP libraries
        words = command.lower().split()

        # Extract action
        actions = ['pick', 'grasp', 'move', 'navigate', 'go', 'put', 'place', 'follow', 'stop']
        action = None
        for word in words:
            if word in actions:
                action = word
                break

        # Extract object
        objects = ['cup', 'box', 'chair', 'table', 'ball', 'book', 'person']
        target_object = None
        for word in words:
            if word in objects:
                target_object = word
                break

        # Extract location
        locations = ['kitchen', 'living room', 'bedroom', 'office', 'table', 'shelf', 'counter']
        target_location = None
        for word in words:
            if word in locations:
                target_location = word
                break

        return {
            'action': action,
            'target_object': target_object,
            'target_location': target_location,
            'original_command': command
        }

class ActionService(VLAMicroservice):
    """Action generation microservice"""
    def __init__(self, config: Dict[str, any]):
        super().__init__('action_service', config)
        self.action_model = None
        self.controller = None

    async def initialize(self):
        """Initialize action service"""
        model_path = self.config.get('action_model_path', 'models/action_model.pt')
        self.action_model = torch.load(model_path)
        self.action_model.eval()

        # Initialize robot controller
        self.controller = self.create_robot_controller()

    async def process_request(self, request_data: Dict[str, any]) -> Dict[str, any]:
        """Process action request"""
        if not all(key in request_data for key in ['vision_features', 'language_features']):
            return {'error': 'Missing required features'}

        vision_features = torch.tensor(request_data['vision_features'])
        language_features = torch.tensor(request_data['language_features'])

        with torch.no_grad():
            action_prediction = self.action_model(vision_features, language_features)

        # Convert to action command
        action_command = self.convert_to_action_command(action_prediction)

        return {
            'action_prediction': action_prediction.cpu().numpy().tolist(),
            'action_command': action_command,
            'timestamp': time.time()
        }

    def convert_to_action_command(self, action_prediction: torch.Tensor) -> Dict[str, any]:
        """Convert action prediction to robot command"""
        # This would convert continuous action space to robot-specific commands
        # For humanoid robots, this might include joint angles, velocities, etc.
        action_vector = action_prediction.cpu().numpy()

        # Example: convert to joint commands for humanoid robot
        joint_commands = {
            'head_yaw': action_vector[0],
            'head_pitch': action_vector[1],
            'left_shoulder_pitch': action_vector[2],
            'left_shoulder_roll': action_vector[3],
            'left_elbow': action_vector[4],
            'right_shoulder_pitch': action_vector[5],
            'right_shoulder_roll': action_vector[6],
            'right_elbow': action_vector[7],
            'left_hip_pitch': action_vector[8],
            'left_hip_roll': action_vector[9],
            'left_knee': action_vector[10],
            'right_hip_pitch': action_vector[11],
            'right_hip_roll': action_vector[12],
            'right_knee': action_vector[13]
        }

        return {
            'joint_commands': joint_commands,
            'command_type': 'joint_position',
            'execution_time': 0.1  # Estimated execution time
        }

class VLACoordinator:
    """Coordinate VLA microservices"""
    def __init__(self, service_configs: Dict[str, Dict[str, any]]):
        self.service_configs = service_configs
        self.services = {}
        self.communication_layer = CommunicationLayer()

    async def initialize_services(self):
        """Initialize all VLA services"""
        # Initialize vision service
        vision_service = VisionService(self.service_configs.get('vision', {}))
        await vision_service.start_service()
        self.services['vision'] = vision_service

        # Initialize language service
        language_service = LanguageService(self.service_configs.get('language', {}))
        await language_service.start_service()
        self.services['language'] = language_service

        # Initialize action service
        action_service = ActionService(self.service_configs.get('action', {}))
        await action_service.start_service()
        self.services['action'] = action_service

    async def process_vla_request(self, images: List[any], commands: List[str]) -> List[Dict[str, any]]:
        """Process VLA request through coordinated services"""
        results = []

        # Process each request in parallel
        tasks = []
        for image, command in zip(images, commands):
            task = asyncio.create_task(self.process_single_request(image, command))
            tasks.append(task)

        # Wait for all results
        individual_results = await asyncio.gather(*tasks)

        return individual_results

    async def process_single_request(self, image: any, command: str) -> Dict[str, any]:
        """Process single VLA request"""
        start_time = time.time()

        # Process vision
        vision_result = await self.services['vision'].process_request({'image': image})
        if 'error' in vision_result:
            return {'error': f"Vision service error: {vision_result['error']}"}

        # Process language
        language_result = await self.services['language'].process_request({'command': command})
        if 'error' in language_result:
            return {'error': f"Language service error: {language_result['error']}"}

        # Process action
        action_input = {
            'vision_features': vision_result['features'],
            'language_features': language_result['embeddings']
        }
        action_result = await self.services['action'].process_request(action_input)
        if 'error' in action_result:
            return {'error': f"Action service error: {action_result['error']}"}

        total_time = time.time() - start_time

        return {
            'vision_result': vision_result,
            'language_result': language_result,
            'action_result': action_result,
            'total_processing_time': total_time,
            'timestamp': time.time()
        }

    async def get_system_health(self) -> Dict[str, any]:
        """Get health status of all services"""
        health_status = {}
        for service_name, service in self.services.items():
            health_status[service_name] = service.get_health_status()

        return {
            'overall_status': self.calculate_overall_health(health_status),
            'individual_services': health_status,
            'timestamp': time.time()
        }

    def calculate_overall_health(self, health_status: Dict[str, any]) -> str:
        """Calculate overall system health"""
        service_statuses = [status['status'] for status in health_status.values()]

        if 'error' in service_statuses:
            return 'error'
        elif 'warning' in service_statuses:
            return 'warning'
        elif all(status == 'running' for status in service_statuses):
            return 'healthy'
        else:
            return 'degraded'

    async def shutdown_services(self):
        """Shutdown all services"""
        for service in self.services.values():
            await service.stop_service()

class CommunicationLayer:
    """Handle communication between microservices"""
    def __init__(self):
        self.message_queue = asyncio.Queue()
        self.service_endpoints = {}
        self.connection_pool = {}

    async def send_message(self, service_name: str, message: Dict[str, any]) -> Optional[Dict[str, any]]:
        """Send message to service"""
        if service_name not in self.service_endpoints:
            return {'error': f'Service {service_name} not available'}

        endpoint = self.service_endpoints[service_name]
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    endpoint,
                    json=message,
                    timeout=aiohttp.ClientTimeout(total=10.0)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result
                    else:
                        error_text = await response.text()
                        return {'error': f'Service returned {response.status}: {error_text}'}
        except asyncio.TimeoutError:
            return {'error': 'Service request timed out'}
        except Exception as e:
            return {'error': f'Communication error: {str(e)}'}

    async def broadcast_message(self, message: Dict[str, any],
                              service_names: List[str]) -> Dict[str, any]:
        """Broadcast message to multiple services"""
        tasks = []
        for service_name in service_names:
            task = asyncio.create_task(self.send_message(service_name, message))
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        response = {}
        for service_name, result in zip(service_names, results):
            if isinstance(result, Exception):
                response[service_name] = {'error': str(result)}
            else:
                response[service_name] = result

        return response

    def register_service(self, service_name: str, endpoint: str):
        """Register service endpoint"""
        self.service_endpoints[service_name] = endpoint

    def deregister_service(self, service_name: str):
        """Deregister service endpoint"""
        if service_name in self.service_endpoints:
            del self.service_endpoints[service_name]

# Example usage of microservices architecture
async def deploy_vla_microservices():
    """Deploy VLA system using microservices architecture"""
    # Define service configurations
    service_configs = {
        'vision': {
            'model_path': 'models/vision_model.pt',
            'input_size': [224, 224],
            'batch_size': 1
        },
        'language': {
            'model_name': 'bert-base-uncased',
            'max_length': 128,
            'batch_size': 1
        },
        'action': {
            'model_path': 'models/action_model.pt',
            'action_space': 14,  # 14 DoF for humanoid
            'control_frequency': 100  # 100 Hz
        }
    }

    # Initialize coordinator
    coordinator = VLACoordinator(service_configs)

    # Start services
    await coordinator.initialize_services()

    # Example usage
    images = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)]  # Mock image
    commands = ["Pick up the red cup"]

    results = await coordinator.process_vla_request(images, commands)
    print(f"VLA results: {results}")

    # Get system health
    health = await coordinator.get_system_health()
    print(f"System health: {health}")

    # Shutdown services
    await coordinator.shutdown_services()

    return results
```

### 2. Containerized Deployment

```yaml
# Docker Compose for VLA microservices
version: '3.8'

services:
  vla-api-gateway:
    build:
      context: .
      dockerfile: Dockerfile.api
    ports:
      - "8080:8080"
    environment:
      - VISION_SERVICE_URL=http://vision-service:8001
      - LANGUAGE_SERVICE_URL=http://language-service:8002
      - ACTION_SERVICE_URL=http://action-service:8003
    depends_on:
      - vision-service
      - language-service
      - action-service
    volumes:
      - ./models:/app/models
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '2'
        reservations:
          memory: 1G
          cpus: '1'

  vision-service:
    build:
      context: .
      dockerfile: Dockerfile.vision
    ports:
      - "8001:8001"
    environment:
      - MODEL_PATH=/models/vision_model.pt
      - GPU_ENABLED=true
    volumes:
      - ./models:/models
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '4'
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
        reservations:
          memory: 2G
          cpus: '2'

  language-service:
    build:
      context: .
      dockerfile: Dockerfile.language
    ports:
      - "8002:8002"
    environment:
      - MODEL_NAME=bert-base-uncased
      - MAX_LENGTH=128
    volumes:
      - ./models:/models
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '2'
        reservations:
          memory: 1G
          cpus: '1'

  action-service:
    build:
      context: .
      dockerfile: Dockerfile.action
    ports:
      - "8003:8003"
    environment:
      - MODEL_PATH=/models/action_model.pt
      - CONTROL_FREQUENCY=100
    volumes:
      - ./models:/models
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '2'
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
        reservations:
          memory: 1G
          cpus: '1'

  monitoring:
    image: grafana/grafana
    ports:
      - "3000:3000"
    volumes:
      - grafana-storage:/var/lib/grafana
    depends_on:
      - prometheus

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

volumes:
  grafana-storage:
```

```dockerfile
# Dockerfile for VLA API Gateway
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/api_gateway/ /app/

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["python", "-m", "api_gateway.main"]
```

```python
# API Gateway implementation
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import asyncio
import aiohttp
import time

app = FastAPI(title="VLA API Gateway", version="1.0.0")

class VLARequest(BaseModel):
    images: List[str]  # Base64 encoded images or URLs
    commands: List[str]
    metadata: Dict[str, Any] = {}

class VLEResponse(BaseModel):
    results: List[Dict[str, Any]]
    processing_time: float
    timestamp: float

@app.post("/vla/inference", response_model=VLEResponse)
async def vla_inference(request: VLARequest):
    """Process VLA inference request"""
    start_time = time.time()

    # Validate request
    if len(request.images) != len(request.commands):
        raise HTTPException(status_code=400, detail="Images and commands must have same length")

    # Process through microservices
    vision_results = await call_vision_service(request.images)
    language_results = await call_language_service(request.commands)

    # Combine results and generate actions
    action_results = await call_action_service(vision_results, language_results)

    # Combine all results
    results = []
    for v_res, l_res, a_res in zip(vision_results, language_results, action_results):
        results.append({
            'vision': v_res,
            'language': l_res,
            'action': a_res
        })

    processing_time = time.time() - start_time

    return VLEResponse(
        results=results,
        processing_time=processing_time,
        timestamp=time.time()
    )

async def call_vision_service(images: List[str]) -> List[Dict[str, Any]]:
    """Call vision service"""
    async with aiohttp.ClientSession() as session:
        tasks = []
        for image in images:
            task = call_single_vision_service(session, image)
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        return results

async def call_single_vision_service(session: aiohttp.ClientSession, image: str) -> Dict[str, Any]:
    """Call single vision service request"""
    vision_url = "http://vision-service:8001/process"
    payload = {'image': image}

    try:
        async with session.post(vision_url, json=payload, timeout=10.0) as response:
            if response.status == 200:
                return await response.json()
            else:
                raise HTTPException(status_code=response.status, detail="Vision service error")
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Vision service timeout")

async def call_language_service(commands: List[str]) -> List[Dict[str, Any]]:
    """Call language service"""
    async with aiohttp.ClientSession() as session:
        tasks = []
        for command in commands:
            task = call_single_language_service(session, command)
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        return results

async def call_single_language_service(session: aiohttp.ClientSession, command: str) -> Dict[str, Any]:
    """Call single language service request"""
    language_url = "http://language-service:8002/process"
    payload = {'command': command}

    try:
        async with session.post(language_url, json=payload, timeout=10.0) as response:
            if response.status == 200:
                return await response.json()
            else:
                raise HTTPException(status_code=response.status, detail="Language service error")
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Language service timeout")

async def call_action_service(vision_results: List[Dict[str, Any]],
                            language_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Call action service with combined features"""
    async with aiohttp.ClientSession() as session:
        tasks = []
        for v_res, l_res in zip(vision_results, language_results):
            task = call_single_action_service(session, v_res, l_res)
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        return results

async def call_single_action_service(session: aiohttp.ClientSession,
                                   vision_result: Dict[str, Any],
                                   language_result: Dict[str, Any]) -> Dict[str, Any]:
    """Call single action service request"""
    action_url = "http://action-service:8003/generate"
    payload = {
        'vision_features': vision_result.get('features', []),
        'language_features': language_result.get('embeddings', [])
    }

    try:
        async with session.post(action_url, json=payload, timeout=10.0) as response:
            if response.status == 200:
                return await response.json()
            else:
                raise HTTPException(status_code=response.status, detail="Action service error")
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Action service timeout")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "services": {
            "vision": "connected",
            "language": "connected",
            "action": "connected"
        }
    }

@app.get("/metrics")
async def get_metrics():
    """Get system metrics"""
    # This would return detailed metrics about the system
    return {
        "requests_processed": 0,
        "average_response_time": 0.0,
        "error_rate": 0.0,
        "service_health": {
            "vision_service": {"latency": 0.02, "availability": 1.0},
            "language_service": {"latency": 0.015, "availability": 1.0},
            "action_service": {"latency": 0.025, "availability": 1.0}
        }
    }
```

## Deployment Monitoring and Management

### 1. System Monitoring

```python
# System monitoring for VLA deployments
import psutil
import GPUtil
import time
from datetime import datetime
import json
import os

class SystemMonitor:
    """Monitor system resources and performance"""
    def __init__(self, log_directory="logs"):
        self.log_directory = log_directory
        self.monitoring_data = []
        self.start_time = time.time()

        # Create log directory if it doesn't exist
        os.makedirs(log_directory, exist_ok=True)

    def collect_system_metrics(self) -> Dict[str, any]:
        """Collect comprehensive system metrics"""
        metrics = {
            'timestamp': time.time(),
            'cpu': {
                'usage_percent': psutil.cpu_percent(interval=1),
                'load_average': psutil.getloadavg(),
                'memory_percent': psutil.virtual_memory().percent,
                'memory_available_gb': psutil.virtual_memory().available / (1024**3),
                'memory_total_gb': psutil.virtual_memory().total / (1024**3)
            },
            'gpu': self.get_gpu_metrics(),
            'disk': {
                'usage_percent': psutil.disk_usage('/').percent,
                'free_gb': psutil.disk_usage('/').free / (1024**3),
                'total_gb': psutil.disk_usage('/').total / (1024**3)
            },
            'network': self.get_network_metrics(),
            'processes': self.get_process_metrics(),
            'uptime_seconds': time.time() - self.start_time
        }

        return metrics

    def get_gpu_metrics(self) -> Dict[str, any]:
        """Get GPU metrics if available"""
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]  # Primary GPU
            return {
                'count': len(gpus),
                'utilization_percent': gpu.load * 100,
                'memory_used_mb': gpu.memoryUsed,
                'memory_total_mb': gpu.memoryTotal,
                'memory_utilization_percent': gpu.memoryUtil * 100,
                'temperature_celsius': gpu.temperature,
                'power_draw_watts': gpu.powerDraw if hasattr(gpu, 'powerDraw') else 0
            }
        else:
            return {
                'count': 0,
                'utilization_percent': 0,
                'memory_used_mb': 0,
                'memory_total_mb': 0,
                'memory_utilization_percent': 0,
                'temperature_celsius': 0,
                'power_draw_watts': 0
            }

    def get_network_metrics(self) -> Dict[str, any]:
        """Get network metrics"""
        net_io = psutil.net_io_counters()
        return {
            'bytes_sent': net_io.bytes_sent,
            'bytes_recv': net_io.bytes_recv,
            'packets_sent': net_io.packets_sent,
            'packets_recv': net_io.packets_recv,
            'current_upload_bps': net_io.bytes_sent,
            'current_download_bps': net_io.bytes_recv
        }

    def get_process_metrics(self) -> Dict[str, any]:
        """Get metrics for VLA-related processes"""
        vla_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'memory_info']):
            try:
                if any(keyword in proc.info['name'].lower() for keyword in ['python', 'torch', 'cuda', 'robot']):
                    vla_processes.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        return {
            'count': len(vla_processes),
            'processes': vla_processes
        }

    def log_metrics(self, metrics: Dict[str, any]):
        """Log metrics to file"""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = f"{self.log_directory}/system_metrics_{timestamp}.json"

        with open(log_file, 'w') as f:
            json.dump(metrics, f, indent=2)

        # Also append to ongoing monitoring data
        self.monitoring_data.append(metrics)

        # Keep only recent data (last 1000 entries)
        if len(self.monitoring_data) > 1000:
            self.monitoring_data = self.monitoring_data[-1000:]

    def get_performance_alerts(self, metrics: Dict[str, any]) -> List[str]:
        """Get performance alerts based on metrics"""
        alerts = []

        # CPU alerts
        if metrics['cpu']['usage_percent'] > 90:
            alerts.append(f"High CPU usage: {metrics['cpu']['usage_percent']:.1f}%")

        if metrics['cpu']['memory_percent'] > 85:
            alerts.append(f"High memory usage: {metrics['cpu']['memory_percent']:.1f}%")

        # GPU alerts
        if metrics['gpu']['utilization_percent'] > 95:
            alerts.append(f"High GPU utilization: {metrics['gpu']['utilization_percent']:.1f}%")

        if metrics['gpu']['memory_utilization_percent'] > 90:
            alerts.append(f"High GPU memory usage: {metrics['gpu']['memory_utilization_percent']:.1f}%")

        if metrics['gpu']['temperature_celsius'] > 80:
            alerts.append(f"High GPU temperature: {metrics['gpu']['temperature_celsius']:.1f}°C")

        # Disk alerts
        if metrics['disk']['usage_percent'] > 90:
            alerts.append(f"High disk usage: {metrics['disk']['usage_percent']:.1f}%")

        return alerts

    def generate_performance_report(self) -> str:
        """Generate performance report from collected metrics"""
        if not self.monitoring_data:
            return "No monitoring data available"

        # Calculate averages
        cpu_usage_avg = np.mean([m['cpu']['usage_percent'] for m in self.monitoring_data])
        memory_usage_avg = np.mean([m['cpu']['memory_percent'] for m in self.monitoring_data])
        gpu_usage_avg = np.mean([m['gpu']['utilization_percent'] for m in self.monitoring_data if m['gpu']['count'] > 0])
        disk_usage_avg = np.mean([m['disk']['usage_percent'] for m in self.monitoring_data])

        # Find peaks
        cpu_usage_peak = max(m['cpu']['usage_percent'] for m in self.monitoring_data)
        gpu_usage_peak = max(m['gpu']['utilization_percent'] for m in self.monitoring_data if m['gpu']['count'] > 0)

        report = f"""
# VLA System Performance Report

## Overview
- **Monitoring Duration**: {(time.time() - self.start_time) / 3600:.2f} hours
- **Total Samples**: {len(self.monitoring_data)}
- **Report Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Average Resource Usage
- **CPU Usage**: {cpu_usage_avg:.2f}%
- **Memory Usage**: {memory_usage_avg:.2f}%
- **GPU Usage**: {gpu_usage_avg:.2f}% (if available)
- **Disk Usage**: {disk_usage_avg:.2f}%

## Peak Usage
- **Peak CPU Usage**: {cpu_usage_peak:.2f}%
- **Peak GPU Usage**: {gpu_usage_peak:.2f}% (if available)

## Recommendations
"""
        if cpu_usage_avg > 80:
            report += "- Consider optimizing CPU-intensive operations\n"
        if memory_usage_avg > 85:
            report += "- Investigate memory leaks or optimize memory usage\n"
        if gpu_usage_avg > 85:
            report += "- Consider model optimization for GPU efficiency\n"
        if disk_usage_avg > 90:
            report += "- Clean up disk space or increase storage capacity\n"

        if not any(cpu_usage_avg > 80, memory_usage_avg > 85, gpu_usage_avg > 85, disk_usage_avg > 90):
            report += "- System performance is within acceptable ranges\n"

        return report.strip()

    def save_performance_report(self, filename: str = None):
        """Save performance report to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.log_directory}/performance_report_{timestamp}.txt"

        report = self.generate_performance_report()

        with open(filename, 'w') as f:
            f.write(report)

        print(f"Performance report saved to {filename}")
        return filename

class DeploymentManager:
    """Manage VLA system deployment"""
    def __init__(self, config_file: str = "deployment_config.json"):
        self.config_file = config_file
        self.deployment_config = self.load_deployment_config()
        self.system_monitor = SystemMonitor()
        self.health_checker = HealthChecker()

    def load_deployment_config(self) -> Dict[str, any]:
        """Load deployment configuration"""
        try:
            with open(self.config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Create default configuration
            default_config = {
                "services": {
                    "vision": {
                        "model_path": "models/vision_model.pt",
                        "gpu_enabled": True,
                        "batch_size": 1,
                        "max_workers": 2
                    },
                    "language": {
                        "model_name": "bert-base-uncased",
                        "max_length": 128,
                        "batch_size": 1,
                        "max_workers": 1
                    },
                    "action": {
                        "model_path": "models/action_model.pt",
                        "control_frequency": 100,
                        "max_workers": 1
                    }
                },
                "resources": {
                    "cpu_limit": "4",
                    "memory_limit": "8G",
                    "gpu_limit": 1
                },
                "monitoring": {
                    "enabled": True,
                    "interval_seconds": 5,
                    "log_directory": "logs"
                },
                "security": {
                    "api_key_required": True,
                    "ssl_enabled": True,
                    "rate_limiting": 100  # requests per minute
                }
            }

            with open(self.config_file, 'w') as f:
                json.dump(default_config, f, indent=2)

            return default_config

    def deploy_services(self):
        """Deploy VLA services according to configuration"""
        print("Starting VLA system deployment...")

        # Start monitoring
        if self.deployment_config['monitoring']['enabled']:
            self.start_monitoring()

        # Deploy vision service
        self.deploy_vision_service()

        # Deploy language service
        self.deploy_language_service()

        # Deploy action service
        self.deploy_action_service()

        # Deploy API gateway
        self.deploy_api_gateway()

        print("VLA system deployment completed successfully!")

    def deploy_vision_service(self):
        """Deploy vision service"""
        vision_config = self.deployment_config['services']['vision']

        # This would actually deploy the service
        # For now, just print the configuration
        print(f"Deploying vision service with config: {vision_config}")

        # In a real deployment, this would:
        # - Load the vision model
        # - Start the vision service
        # - Configure GPU resources
        # - Set up health checks

    def deploy_language_service(self):
        """Deploy language service"""
        language_config = self.deployment_config['services']['language']

        print(f"Deploying language service with config: {language_config}")

    def deploy_action_service(self):
        """Deploy action service"""
        action_config = self.deployment_config['services']['action']

        print(f"Deploying action service with config: {action_config}")

    def deploy_api_gateway(self):
        """Deploy API gateway"""
        print("Deploying API gateway...")

    def start_monitoring(self):
        """Start system monitoring"""
        monitoring_interval = self.deployment_config['monitoring']['interval_seconds']
        log_dir = self.deployment_config['monitoring']['log_directory']

        self.system_monitor = SystemMonitor(log_dir)

        # Start monitoring loop in background
        import threading
        monitoring_thread = threading.Thread(target=self.monitoring_loop, args=(monitoring_interval,))
        monitoring_thread.daemon = True
        monitoring_thread.start()

        print(f"Started system monitoring (interval: {monitoring_interval}s)")

    def monitoring_loop(self, interval: int):
        """Continuous monitoring loop"""
        while True:
            try:
                # Collect metrics
                metrics = self.system_monitor.collect_system_metrics()

                # Log metrics
                self.system_monitor.log_metrics(metrics)

                # Check for alerts
                alerts = self.system_monitor.get_performance_alerts(metrics)
                if alerts:
                    print("PERFORMANCE ALERTS:")
                    for alert in alerts:
                        print(f"  - {alert}")

                # Sleep for specified interval
                time.sleep(interval)

            except KeyboardInterrupt:
                print("Monitoring stopped by user")
                break
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(interval)  # Continue monitoring despite error

    def check_system_health(self) -> Dict[str, any]:
        """Check overall system health"""
        health_status = {
            'timestamp': time.time(),
            'services_healthy': self.health_checker.check_all_services(),
            'system_resources': self.system_monitor.collect_system_metrics(),
            'performance_score': self.calculate_performance_score(),
            'recommendations': self.get_health_recommendations()
        }

        return health_status

    def calculate_performance_score(self) -> float:
        """Calculate overall system performance score (0-1)"""
        # Get latest metrics
        if not self.system_monitor.monitoring_data:
            return 0.5  # Default score if no data

        latest_metrics = self.system_monitor.monitoring_data[-1]

        # Calculate score based on resource usage
        cpu_score = max(0, 1 - (latest_metrics['cpu']['usage_percent'] / 100))
        memory_score = max(0, 1 - (latest_metrics['cpu']['memory_percent'] / 100))
        gpu_score = max(0, 1 - (latest_metrics['gpu']['utilization_percent'] / 100)) if latest_metrics['gpu']['count'] > 0 else 1.0

        # Weighted average (GPU-heavy systems get more GPU weight)
        weights = {
            'cpu': 0.3,
            'memory': 0.4,
            'gpu': 0.3
        }

        performance_score = (
            weights['cpu'] * cpu_score +
            weights['memory'] * memory_score +
            weights['gpu'] * gpu_score
        )

        return performance_score

    def get_health_recommendations(self) -> List[str]:
        """Get health-based recommendations"""
        recommendations = []

        # Check if system needs optimization
        if self.calculate_performance_score() < 0.7:
            recommendations.append("System performance below threshold - consider optimization")

        # Check resource usage
        latest_metrics = self.system_monitor.monitoring_data[-1] if self.system_monitor.monitoring_data else {}

        if latest_metrics.get('cpu', {}).get('memory_percent', 0) > 90:
            recommendations.append("High memory usage detected - consider memory optimization")

        if latest_metrics.get('gpu', {}).get('utilization_percent', 0) > 95:
            recommendations.append("High GPU utilization - consider model optimization")

        if not recommendations:
            recommendations.append("System health is good - no immediate recommendations")

        return recommendations

    def undeploy_services(self):
        """Undeploy all services"""
        print("Starting VLA system undeployment...")

        # Stop monitoring
        # This would actually stop the monitoring thread

        # Stop services in reverse order
        self.stop_api_gateway()
        self.stop_action_service()
        self.stop_language_service()
        self.stop_vision_service()

        print("VLA system undeployment completed!")

    def stop_vision_service(self):
        """Stop vision service"""
        print("Stopping vision service...")

    def stop_language_service(self):
        """Stop language service"""
        print("Stopping language service...")

    def stop_action_service(self):
        """Stop action service"""
        print("Stopping action service...")

    def stop_api_gateway(self):
        """Stop API gateway"""
        print("Stopping API gateway...")

    def restart_service(self, service_name: str):
        """Restart a specific service"""
        print(f"Restarting {service_name} service...")

        # Stop the service
        if service_name == 'vision':
            self.stop_vision_service()
        elif service_name == 'language':
            self.stop_language_service()
        elif service_name == 'action':
            self.stop_action_service()
        elif service_name == 'gateway':
            self.stop_api_gateway()
        else:
            print(f"Unknown service: {service_name}")
            return

        # Start the service again
        if service_name == 'vision':
            self.deploy_vision_service()
        elif service_name == 'language':
            self.deploy_language_service()
        elif service_name == 'action':
            self.deploy_action_service()
        elif service_name == 'gateway':
            self.deploy_api_gateway()

        print(f"{service_name} service restarted successfully")

class HealthChecker:
    """Check health of VLA services"""
    def __init__(self):
        self.service_endpoints = {
            'vision': 'http://localhost:8001/health',
            'language': 'http://localhost:8002/health',
            'action': 'http://localhost:8003/health',
            'gateway': 'http://localhost:8080/health'
        }

    async def check_service_health(self, service_name: str) -> bool:
        """Check health of a specific service"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.service_endpoints[service_name], timeout=5.0) as response:
                    return response.status == 200
        except:
            return False

    async def check_all_services(self) -> Dict[str, bool]:
        """Check health of all services"""
        tasks = []
        for service_name in self.service_endpoints.keys():
            task = asyncio.create_task(self.check_service_health(service_name))
            tasks.append(task)

        results = await asyncio.gather(*tasks)

        health_status = {}
        for service_name, is_healthy in zip(self.service_endpoints.keys(), results):
            health_status[service_name] = is_healthy

        return health_status

    def get_system_health_summary(self, health_status: Dict[str, bool]) -> str:
        """Get system health summary"""
        healthy_services = sum(1 for status in health_status.values() if status)
        total_services = len(health_status)

        if healthy_services == total_services:
            return "All services healthy"
        elif healthy_services == 0:
            return "All services unhealthy"
        else:
            unhealthy_services = [name for name, status in health_status.items() if not status]
            return f"{healthy_services}/{total_services} services healthy. Unhealthy: {', '.join(unhealthy_services)}"
```

## Deployment Best Practices

### 1. Configuration Management

```python
# Configuration management for VLA deployments
import yaml
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional

@dataclass
class ModelConfig:
    """Configuration for VLA model deployment"""
    model_path: str
    input_size: List[int]
    batch_size: int
    precision: str  # 'fp32', 'fp16', 'int8'
    max_workers: int
    optimization_enabled: bool
    quantization_enabled: bool

@dataclass
class HardwareConfig:
    """Hardware configuration for deployment"""
    platform: str  # 'jetson_orin', 'jetson_xavier', 'desktop_gpu', 'cpu_only'
    gpu_enabled: bool
    memory_limit_gb: float
    cpu_cores: int
    network_bandwidth_mbps: float
    thermal_management: bool

@dataclass
class PerformanceConfig:
    """Performance configuration"""
    target_fps: float
    max_latency_ms: float
    min_accuracy: float
    safety_threshold: float
    backup_strategy: str  # 'local', 'cloud', 'hybrid'

@dataclass
class SecurityConfig:
    """Security configuration"""
    api_key_required: bool
    ssl_enabled: bool
    authentication_enabled: bool
    data_encryption: bool
    access_control: Dict[str, List[str]]

class VLAConfiguration:
    """Complete VLA system configuration"""
    def __init__(self, config_path: str = None):
        if config_path:
            self.load_from_file(config_path)
        else:
            self.model_config = ModelConfig(
                model_path="models/vla_model.pt",
                input_size=[3, 224, 224],
                batch_size=1,
                precision='fp16',
                max_workers=1,
                optimization_enabled=True,
                quantization_enabled=True
            )
            self.hardware_config = HardwareConfig(
                platform='jetson_orin',
                gpu_enabled=True,
                memory_limit_gb=32.0,
                cpu_cores=12,
                network_bandwidth_mbps=1000.0,
                thermal_management=True
            )
            self.performance_config = PerformanceConfig(
                target_fps=30.0,
                max_latency_ms=50.0,
                min_accuracy=0.85,
                safety_threshold=0.95,
                backup_strategy='hybrid'
            )
            self.security_config = SecurityConfig(
                api_key_required=True,
                ssl_enabled=True,
                authentication_enabled=True,
                data_encryption=True,
                access_control={'admin': ['read', 'write', 'execute'], 'user': ['read']}
            )

    def load_from_file(self, config_path: str):
        """Load configuration from file"""
        with open(config_path, 'r') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                config_data = yaml.safe_load(f)
            else:
                config_data = json.load(f)

        self.model_config = ModelConfig(**config_data.get('model', {}))
        self.hardware_config = HardwareConfig(**config_data.get('hardware', {}))
        self.performance_config = PerformanceConfig(**config_data.get('performance', {}))
        self.security_config = SecurityConfig(**config_data.get('security', {}))

    def save_to_file(self, config_path: str):
        """Save configuration to file"""
        config_data = {
            'model': asdict(self.model_config),
            'hardware': asdict(self.hardware_config),
            'performance': asdict(self.performance_config),
            'security': asdict(self.security_config)
        }

        with open(config_path, 'w') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                yaml.dump(config_data, f, default_flow_style=False)
            else:
                json.dump(config_data, f, indent=2)

    def validate_configuration(self) -> List[str]:
        """Validate configuration for deployment"""
        validation_errors = []

        # Validate model configuration
        if not os.path.exists(self.model_config.model_path):
            validation_errors.append(f"Model path does not exist: {self.model_config.model_path}")

        if self.model_config.precision not in ['fp32', 'fp16', 'int8']:
            validation_errors.append(f"Invalid precision: {self.model_config.precision}")

        # Validate hardware configuration
        if self.hardware_config.memory_limit_gb < 2.0:
            validation_errors.append("Insufficient memory limit (< 2GB)")

        if self.hardware_config.cpu_cores < 1:
            validation_errors.append("Invalid CPU cores configuration")

        # Validate performance configuration
        if self.performance_config.target_fps <= 0:
            validation_errors.append("Invalid target FPS")

        if self.performance_config.max_latency_ms <= 0:
            validation_errors.append("Invalid max latency")

        # Validate security configuration
        if self.security_config.api_key_required and not os.environ.get('VLA_API_KEY'):
            validation_errors.append("API key required but not set in environment")

        return validation_errors

    def get_optimization_recommendations(self) -> List[str]:
        """Get optimization recommendations based on configuration"""
        recommendations = []

        # Memory optimization recommendations
        if self.hardware_config.memory_limit_gb < 8.0:
            recommendations.append("Consider enabling model quantization for memory-constrained deployment")
            recommendations.append("Reduce batch size to conserve memory")

        if self.hardware_config.memory_limit_gb > 32.0:
            recommendations.append("Consider increasing batch size for better throughput")

        # Performance recommendations
        if self.performance_config.target_fps > 60:
            recommendations.append("High FPS target may require specialized hardware optimization")

        if self.hardware_config.platform == 'cpu_only':
            recommendations.append("CPU-only deployment may not meet real-time requirements")

        if self.hardware_config.gpu_enabled and self.model_config.precision == 'fp16':
            recommendations.append("FP16 precision recommended for GPU deployment")

        # Security recommendations
        if not self.security_config.ssl_enabled:
            recommendations.append("SSL recommended for production deployment")

        return recommendations

    def apply_optimizations(self, model: nn.Module) -> nn.Module:
        """Apply optimizations based on configuration"""
        optimized_model = model

        # Apply quantization if enabled
        if self.model_config.quantization_enabled and self.model_config.precision == 'int8':
            optimized_model = self.apply_int8_quantization(optimized_model)

        # Apply tensor optimization if using GPU
        if self.hardware_config.gpu_enabled and self.model_config.precision == 'fp16':
            optimized_model = self.apply_fp16_optimization(optimized_model)

        # Apply model optimization
        if self.model_config.optimization_enabled:
            optimized_model = self.apply_model_optimization(optimized_model)

        return optimized_model

    def apply_int8_quantization(self, model: nn.Module) -> nn.Module:
        """Apply INT8 quantization to model"""
        model.eval()
        model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
        torch.quantization.prepare(model, inplace=True)
        torch.quantization.convert(model, inplace=True)
        return model

    def apply_fp16_optimization(self, model: nn.Module) -> nn.Module:
        """Apply FP16 optimization to model"""
        return model.half()

    def apply_model_optimization(self, model: nn.Module) -> nn.Module:
        """Apply general model optimizations"""
        # Optimize for inference
        model.eval()
        optimized_model = torch.jit.optimize_for_inference(torch.jit.script(model))
        return optimized_model

class DeploymentValidator:
    """Validate deployment readiness"""
    def __init__(self, config: VLAConfiguration):
        self.config = config

    def validate_deployment_readiness(self) -> Dict[str, any]:
        """Validate if system is ready for deployment"""
        validation_results = {
            'system_compatibility': self.check_system_compatibility(),
            'resource_availability': self.check_resource_availability(),
            'model_compatibility': self.check_model_compatibility(),
            'security_compliance': self.check_security_compliance(),
            'network_readiness': self.check_network_readiness()
        }

        # Overall readiness score
        total_checks = len(validation_results)
        passed_checks = sum(1 for result in validation_results.values() if result.get('passed', False))
        readiness_score = passed_checks / total_checks if total_checks > 0 else 0.0

        validation_results['readiness_score'] = readiness_score
        validation_results['deployment_ready'] = readiness_score >= 0.8  # 80% threshold

        return validation_results

    def check_system_compatibility(self) -> Dict[str, any]:
        """Check system compatibility with configuration"""
        try:
            import torch
            import torchvision

            # Check PyTorch version compatibility
            torch_version = torch.__version__
            if tuple(map(int, torch_version.split('.')[:2])) < (1, 12):
                return {'passed': False, 'issue': f'PyTorch version {torch_version} too old, 1.12+ required'}

            # Check CUDA availability if GPU requested
            if self.config.hardware_config.gpu_enabled and not torch.cuda.is_available():
                return {'passed': False, 'issue': 'CUDA not available but GPU requested'}

            # Check for required packages
            required_packages = ['transformers', 'numpy', 'opencv-python']
            missing_packages = []
            for package in required_packages:
                try:
                    __import__(package)
                except ImportError:
                    missing_packages.append(package)

            if missing_packages:
                return {'passed': False, 'issue': f'Missing required packages: {missing_packages}'}

            return {'passed': True, 'details': f'PyTorch {torch_version} available'}

        except Exception as e:
            return {'passed': False, 'issue': f'System compatibility check failed: {str(e)}'}

    def check_resource_availability(self) -> Dict[str, any]:
        """Check if system resources are available"""
        import psutil

        # Check memory
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        if available_memory_gb < self.config.hardware_config.memory_limit_gb * 0.8:  # 80% of limit
            return {
                'passed': False,
                'issue': f'Insufficient memory: {available_memory_gb:.2f}GB available, {self.config.hardware_config.memory_limit_gb * 0.8:.2f}GB required'
            }

        # Check CPU cores
        cpu_count = psutil.cpu_count()
        if cpu_count < self.config.hardware_config.cpu_cores:
            return {
                'passed': False,
                'issue': f'Insufficient CPU cores: {cpu_count} available, {self.config.hardware_config.cpu_cores} required'
            }

        # Check disk space
        disk_free_gb = psutil.disk_usage('/').free / (1024**3)
        if disk_free_gb < 5.0:  # Require at least 5GB free
            return {
                'passed': False,
                'issue': f'Insufficient disk space: {disk_free_gb:.2f}GB free, 5GB minimum required'
            }

        return {
            'passed': True,
            'details': {
                'memory_available_gb': available_memory_gb,
                'cpu_cores_available': cpu_count,
                'disk_space_gb': disk_free_gb
            }
        }

    def check_model_compatibility(self) -> Dict[str, any]:
        """Check if model is compatible with hardware"""
        try:
            model_path = self.config.model_config.model_path
            if not os.path.exists(model_path):
                return {'passed': False, 'issue': f'Model file not found: {model_path}'}

            # Load model and check size
            model_size_mb = os.path.getsize(model_path) / (1024 * 1024)

            # Estimate memory requirements
            # This is a rough estimate - in practice, you'd measure actual requirements
            estimated_memory_mb = model_size_mb * 3  # Include activations and buffers

            if estimated_memory_mb > self.config.hardware_config.memory_limit_gb * 1024:
                return {
                    'passed': False,
                    'issue': f'Model too large: {model_size_mb:.2f}MB estimated memory requirement exceeds {self.config.hardware_config.memory_limit_gb}GB limit'
                }

            return {
                'passed': True,
                'details': {
                    'model_size_mb': model_size_mb,
                    'estimated_memory_mb': estimated_memory_mb
                }
            }

        except Exception as e:
            return {'passed': False, 'issue': f'Model compatibility check failed: {str(e)}'}

    def check_security_compliance(self) -> Dict[str, any]:
        """Check security compliance"""
        security_issues = []

        # Check API key
        if self.config.security_config.api_key_required:
            if not os.environ.get('VLA_API_KEY'):
                security_issues.append('VLA_API_KEY environment variable not set')

        # Check SSL certificate
        if self.config.security_config.ssl_enabled:
            cert_path = os.environ.get('SSL_CERT_PATH')
            key_path = os.environ.get('SSL_KEY_PATH')
            if not cert_path or not key_path:
                security_issues.append('SSL certificate paths not configured')
            elif not os.path.exists(cert_path) or not os.path.exists(key_path):
                security_issues.append('SSL certificate files not found')

        if security_issues:
            return {'passed': False, 'issues': security_issues}
        else:
            return {'passed': True, 'details': 'All security requirements met'}

    def check_network_readiness(self) -> Dict[str, any]:
        """Check network readiness"""
        try:
            import socket

            # Check if network is available
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.connect(("8.8.8.8", 80))
            local_ip = sock.getsockname()[0]
            sock.close()

            # Check bandwidth (simplified)
            # In practice, you'd perform actual bandwidth tests
            required_bandwidth = self.config.hardware_config.network_bandwidth_mbps
            estimated_bandwidth = 100  # Placeholder - would measure actual

            if estimated_bandwidth < required_bandwidth * 0.5:  # 50% of required
                return {
                    'passed': False,
                    'issue': f'Insufficient network bandwidth: {estimated_bandwidth}Mbps available, {required_bandwidth}Mbps required'
                }

            return {
                'passed': True,
                'details': {
                    'local_ip': local_ip,
                    'estimated_bandwidth_mbps': estimated_bandwidth
                }
            }

        except Exception as e:
            return {'passed': False, 'issue': f'Network check failed: {str(e)}'}

# Example deployment script
def deploy_vla_system(config_path: str = "deployment_config.yaml"):
    """Deploy VLA system with validation"""
    # Load configuration
    config = VLAConfiguration(config_path)

    # Validate deployment readiness
    validator = DeploymentValidator(config)
    validation_results = validator.validate_deployment_readiness()

    print("Deployment Validation Results:")
    print(f"Readiness Score: {validation_results['readiness_score']:.2f}")
    print(f"Deployment Ready: {validation_results['deployment_ready']}")

    if not validation_results['deployment_ready']:
        print("\nIssues found:")
        for check_name, result in validation_results.items():
            if isinstance(result, dict) and not result.get('passed', True):
                issue = result.get('issue', result.get('issues', 'Unknown issue'))
                print(f"  - {check_name}: {issue}")

        return False

    # Proceed with deployment
    print("\nStarting deployment...")

    # Initialize deployment manager
    deployment_manager = DeploymentManager(config_path)

    # Deploy services
    deployment_manager.deploy_services()

    # Start monitoring
    print("Deployment completed successfully!")
    return True

if __name__ == "__main__":
    success = deploy_vla_system()
    if not success:
        print("Deployment failed - check validation results above")
        exit(1)
```

## Next Steps

In the next section, we'll explore deployment validation and testing methodologies, learning how to thoroughly test VLA systems in both simulated and real-world environments to ensure they meet performance, safety, and reliability requirements before full deployment.