# AI Image Generator - Separate GPU/CPU Versions

Simple AI image generator using Stable Diffusion with dedicated GPU and CPU versions to avoid dependency conflicts.

## 📁 Structure

```
image-generation/
├── gpu-version/          # GPU-optimized version
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── app.py
│   └── docker-compose.yml
├── cpu-version/          # CPU-only version
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── app.py
│   └── docker-compose.yml
└── README.md
```

## 🚀 Quick Start

### GPU Version (Recommended)
```bash
cd gpu-version
docker-compose up --build
```

### CPU Version (Fallback)
```bash
cd cpu-version
docker-compose up --build
```

## 🎯 Features

- **GPU Version**: CUDA-optimized with xformers and triton for fast generation
- **CPU Version**: CPU-optimized without GPU dependencies
- **Separate Dependencies**: No version conflicts between GPU/CPU packages
- **Web Interface**: Gradio interface with device selection
- **Share Links**: Public URLs for external access

## 📡 Access

Both versions will be available at:
- **Local**: http://localhost:7860
- **Share Link**: Displayed in console output

## 🔧 Manual Build (Alternative)

### GPU Version
```bash
cd gpu-version
docker build -t ai-image-gpu .
docker run --rm -p 7860:7860 ai-image-gpu
```

### CPU Version
```bash
cd cpu-version
docker build -t ai-image-cpu .
docker run --rm -p 7860:7860 ai-image-cpu
```

## 💡 Troubleshooting

- **GPU not detected**: Use CPU version instead
- **Port conflicts**: Change port mapping `-p 7861:7860`
- **Memory issues**: Reduce inference steps in the web interface
- **Dependency conflicts**: Each version has isolated requirements
