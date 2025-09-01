# **Flux.1-dev-4bit**

A quantized version of FLUX.1-dev optimized for efficient image generation with reduced memory footprint through 4-bit quantization. This application provides a Gradio-based interface for generating high-quality images using the FLUX.1-dev model with 4-bit quantization. The quantized model maintains image quality while significantly reducing GPU memory requirements, making it accessible for users with limited hardware resources.

## Features

- **4-bit Quantized Model**: Reduced memory usage while maintaining image quality
- **Interactive Web Interface**: User-friendly Gradio interface for easy image generation
- **Customizable Parameters**: Control guidance scale, inference steps, and random seed
- **GPU Acceleration**: Optimized for CUDA-enabled GPUs
- **Efficient Pipeline**: Streamlined inference process for faster generation

## Requirements

### Hardware
- NVIDIA GPU with CUDA support
- Minimum 8GB GPU memory (4-bit quantization reduces requirements significantly)

### Software Dependencies
```
torch
gradio
spaces
optimum[quanto]
diffusers
transformers
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/PRITHIVSAKTHIUR/Flux.1-dev-4bit.git
cd Flux.1-dev-4bit
```

2. Install required dependencies:
```bash
pip install torch gradio spaces optimum[quanto] diffusers transformers
```

3. Ensure you have the quantized model files in the `FLUX.1-dev-4bit` directory:
   - `text_encoder_2.pt`
   - `transformer.pt`

## Usage

### Running the Application

Start the Gradio interface:
```bash
python app.py
```

The application will launch a web interface where you can:

1. **Enter a Prompt**: Describe the image you want to generate
2. **Adjust Guidance Scale**: Control how closely the model follows your prompt (0.0-10.0)
3. **Set Inference Steps**: Number of denoising steps (1-50, higher values may improve quality)
4. **Specify Seed**: For reproducible results

### Parameters

- **Prompt**: Text description of the desired image
- **Guidance Scale**: Controls prompt adherence (default: 0.0)
- **Number of Inference Steps**: Denoising iterations (default: 4)
- **Seed**: Random seed for reproducible generation (default: 12345)

## Model Details

This implementation uses:
- **Base Model**: black-forest-labs/FLUX.1-dev
- **Text Encoder**: OpenAI CLIP ViT-Large-Patch14
- **Secondary Text Encoder**: T5 (quantized to 4-bit)
- **Transformer**: FLUX Transformer2D (quantized to 4-bit)
- **VAE**: AutoencoderKL from FLUX.1-dev
- **Scheduler**: FlowMatchEulerDiscreteScheduler

## Quantization Benefits

The 4-bit quantization provides:
- **Reduced Memory Usage**: Approximately 4x reduction in model size
- **Faster Loading**: Quicker model initialization
- **Lower Hardware Requirements**: Accessible on consumer GPUs
- **Maintained Quality**: Minimal impact on generation quality

## Technical Implementation

The application loads pre-quantized model components and constructs a custom FLUX pipeline. Key optimizations include:

- Selective quantization of memory-intensive components (transformer and text_encoder_2)
- Efficient model loading with torch.load for quantized components
- GPU memory optimization through proper dtype handling

## Limitations

- Requires CUDA-compatible GPU
- Quantized models may have slight quality differences compared to full precision
- Initial model loading time depends on storage speed
- Limited to single image generation per inference

## License

This project is licensed under the Apache License 2.0. See the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests to improve the application.

## Acknowledgments

- Black Forest Labs for the original FLUX.1-dev model
- Hugging Face for the diffusers and transformers libraries
- Gradio team for the interface framework
- Optimum team for quantization tools
