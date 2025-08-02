"""
Simple AI Image Generator
GPU (default/preferred) and CPU modes available
"""

import gradio as gr
import torch
from PIL import Image
import time
import os
import gc
import gc

# Set HuggingFace cache directory before importing transformers/diffusers
os.environ['HF_HOME'] = '/app/cache'
os.environ['TRANSFORMERS_CACHE'] = '/app/cache'
os.environ['HF_HUB_CACHE'] = '/app/cache'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/root/.cache/huggingface'

# Import AI libraries
try:
    from transformers import CLIPTextModel, CLIPTokenizer
    from diffusers import StableDiffusionPipeline
    print("✅ All required libraries imported successfully")
    print(f"🗂️ HF_HOME: {os.environ.get('HF_HOME', 'Not set')}")
    print(f"🗂️ TRANSFORMERS_CACHE: {os.environ.get('TRANSFORMERS_CACHE', 'Not set')}")
except ImportError as e:
    print(f"❌ Failed to import required libraries: {e}")
    print("Please ensure all dependencies are installed")
    exit(1)

# Check GPU availability
CUDA_AVAILABLE = torch.cuda.is_available()
print(f"🔍 CUDA Available: {CUDA_AVAILABLE}")

if CUDA_AVAILABLE:
    gpu_name = torch.cuda.get_device_name(0)
    print(f"🎮 GPU: {gpu_name}")
else:
    print("💻 Running in CPU mode")

def check_cache_status():
    """Check cache directory status"""
    cache_dirs = ["/app/cache", "/root/.cache/huggingface"]
    
    for cache_dir in cache_dirs:
        print(f"📂 Checking {cache_dir}:")
        if os.path.exists(cache_dir):
            try:
                items = os.listdir(cache_dir)
                print(f"   ✅ Exists with {len(items)} items")
                if items:
                    print(f"   📁 Contents: {items[:5]}{'...' if len(items) > 5 else ''}")
                    
                    # Calculate cache size
                    total_size = 0
                    for root, dirs, files in os.walk(cache_dir):
                        for file in files:
                            try:
                                total_size += os.path.getsize(os.path.join(root, file))
                            except (OSError, FileNotFoundError):
                                pass
                    
                    if total_size > 0:
                        size_gb = total_size / (1024**3)
                        print(f"   📊 Cache size: {size_gb:.2f} GB")
                else:
                    print("   📭 Empty directory")
            except Exception as e:
                print(f"   ❌ Error reading directory: {e}")
        else:
            print("   ❌ Directory does not exist")

# Global pipeline variables
gpu_pipeline = None
cpu_pipeline = None

def load_gpu_pipeline():
    """Load Stable Diffusion pipeline for GPU"""
    global gpu_pipeline
    if gpu_pipeline is None and CUDA_AVAILABLE:
        print("🔄 Loading GPU pipeline...")
        
        # Check cache directories
        cache_dir = "/app/cache"
        print(f"📂 App cache directory: {cache_dir}")
        print(f"📂 Cache dir exists: {os.path.exists(cache_dir)}")
        
        try:
            # Clear CUDA cache first
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print(f"🎮 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB total")
            
            print("� Loading model directly to GPU (avoiding slow transfer)...")
            start_time = time.time()
            
            # Load directly to GPU (remove unsupported torch_device parameter)
            gpu_pipeline = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16,
                safety_checker=None,
                requires_safety_checker=False,
                use_safetensors=True,
                cache_dir=cache_dir,
                low_cpu_mem_usage=False,  # Disable to avoid conflicts
                use_auth_token=False,
                local_files_only=True   # Force use of cached files only
            )
            
            # Move pipeline to GPU after loading
            gpu_pipeline = gpu_pipeline.to("cuda")
            
            load_time = time.time() - start_time
            print(f"✅ Model loaded directly to GPU in {load_time:.1f}s")
            
            print("⚡ Enabling optimizations...")
            # Enable memory-efficient attention
            gpu_pipeline.enable_attention_slicing()
            
            # Try to enable XFormers if available
            if hasattr(gpu_pipeline, 'enable_xformers_memory_efficient_attention'):
                try:
                    gpu_pipeline.enable_xformers_memory_efficient_attention()
                    print("✅ XFormers enabled")
                except Exception as e:
                    print(f"⚠️ XFormers not available: {e}")
            
            total_time = time.time() - start_time
            print(f"✅ GPU pipeline ready in {total_time:.1f}s total")
            
            # Check GPU memory usage
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated(0) / 1024**3
                print(f"🎮 GPU Memory used: {memory_used:.1f}GB")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load directly to GPU: {e}")
            print("🔄 Trying alternative approach...")
            
            # Alternative: Load to CPU first, then move quickly
            try:
                print("📥 Loading to CPU first...")
                gpu_pipeline = StableDiffusionPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",
                    torch_dtype=torch.float32,  # Use float32 for CPU
                    safety_checker=None,
                    requires_safety_checker=False,
                    use_safetensors=True,
                    cache_dir=cache_dir,
                    low_cpu_mem_usage=True,
                    use_auth_token=False,
                    local_files_only=True
                )
                
                print("🔄 Converting to half precision and moving to GPU...")
                # Convert to half precision and move to GPU
                gpu_pipeline = gpu_pipeline.half().to("cuda")
                gpu_pipeline.enable_attention_slicing()
                
                print("✅ GPU pipeline loaded with alternative method")
                return True
                
            except Exception as e2:
                print(f"❌ Alternative method failed: {e2}")
                print("🔄 Final fallback: Allow downloads...")
                
                # Final fallback: allow downloads
                try:
                    gpu_pipeline = StableDiffusionPipeline.from_pretrained(
                        "runwayml/stable-diffusion-v1-5",
                        torch_dtype=torch.float16,
                        safety_checker=None,
                        requires_safety_checker=False,
                        use_safetensors=True,
                        cache_dir=cache_dir,
                        low_cpu_mem_usage=True,
                        use_auth_token=False,
                        local_files_only=False
                    )
                    
                    gpu_pipeline = gpu_pipeline.to("cuda")
                    gpu_pipeline.enable_attention_slicing()
                    
                    print("✅ GPU pipeline loaded with final fallback")
                    return True
                    
                except Exception as e3:
                    print(f"❌ Complete failure: {e3}")
                    import traceback
                    traceback.print_exc()
                    gpu_pipeline = None
                    return False
    return True

def load_cpu_pipeline():
    """Load Stable Diffusion pipeline for CPU"""
    global cpu_pipeline
    if cpu_pipeline is None:
        print("🔄 Loading CPU pipeline...")
        
        cache_dir = "/app/cache"
        print(f"📂 CPU Cache directory: {cache_dir}")
        
        try:
            cpu_pipeline = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float32,
                safety_checker=None,
                requires_safety_checker=False,
                use_safetensors=True,
                cache_dir=cache_dir,
                low_cpu_mem_usage=True,
                local_files_only=False,  # Allow downloading if not cached
                resume_download=True     # Resume interrupted downloads
            )
            cpu_pipeline = cpu_pipeline.to("cpu")
            print("✅ CPU pipeline loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load CPU pipeline: {e}")
            import traceback
            traceback.print_exc()
            # Clear any partial pipeline
            cpu_pipeline = None
            return False
    return True

def generate_image(prompt, device_choice, num_inference_steps, guidance_scale, width, height):
    """Generate image using selected device"""
    
    if not prompt.strip():
        return None, "❌ Please enter a prompt"
    
    start_time = time.time()
    
    # Force garbage collection before generation
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    try:
        if device_choice == "GPU" and CUDA_AVAILABLE:
            # Only load pipeline if not already loaded
            if gpu_pipeline is None:
                print(f"🎮 Loading GPU pipeline for: {prompt}")
                if not load_gpu_pipeline():
                    print("⚠️ GPU pipeline failed, falling back to CPU")
                    device_choice = "CPU"  # Fallback to CPU
                else:
                    print("✅ GPU pipeline ready")
            else:
                print("🎮 Using cached GPU pipeline")
            
            if gpu_pipeline is not None:
                print(f"🎮 Generating with GPU: {prompt}")
                with torch.cuda.amp.autocast():  # Use automatic mixed precision for speed
                    image = gpu_pipeline(
                        prompt,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        height=height,
                        width=width,
                        generator=torch.Generator(device="cuda").manual_seed(42)  # Consistent results
                    ).images[0]
                device_used = "GPU"
        
        if device_choice == "CPU" or not CUDA_AVAILABLE:
            # Only load pipeline if not already loaded
            if cpu_pipeline is None:
                print(f"💻 Loading CPU pipeline for: {prompt}")
                if not load_cpu_pipeline():
                    return None, "❌ Failed to load CPU pipeline"
                else:
                    print("✅ CPU pipeline ready")
            else:
                print("💻 Using cached CPU pipeline")
            
            print(f"💻 Generating with CPU: {prompt}")
            image = cpu_pipeline(
                prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                generator=torch.Generator().manual_seed(42)  # Consistent results
            ).images[0]
            device_used = "CPU"
        
        generation_time = time.time() - start_time
        status = f"✅ Generated using {device_used} in {generation_time:.1f}s"
        
        # Clean up memory after generation
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return image, status
        
    except Exception as e:
        error_time = time.time() - start_time
        error_msg = f"❌ Generation failed after {error_time:.1f}s: {str(e)}"
        print(error_msg)
        return None, error_msg

# Create Gradio interface
def create_interface():
    """Create the Gradio web interface"""
    
    with gr.Blocks(title="AI Image Generator", theme=gr.themes.Soft()) as interface:
        
        gr.Markdown("# 🎨 AI Image Generator")
        gr.Markdown("Generate images using Stable Diffusion with GPU or CPU")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input controls
                prompt_input = gr.Textbox(
                    label="Prompt", 
                    placeholder="Enter your image description...",
                    lines=3
                )
                
                device_choice = gr.Radio(
                    choices=["GPU", "CPU"],
                    value="GPU" if CUDA_AVAILABLE else "CPU",
                    label="Device",
                    info="GPU is preferred when available"
                )
                
                with gr.Row():
                    steps_slider = gr.Slider(
                        minimum=10,
                        maximum=50,
                        value=20,
                        step=1,
                        label="Inference Steps"
                    )
                    
                    guidance_slider = gr.Slider(
                        minimum=1.0,
                        maximum=20.0,
                        value=7.5,
                        step=0.5,
                        label="Guidance Scale"
                    )
                
                width_slider = gr.Slider(
                    minimum=256,
                    maximum=2048,
                    value=512,
                    step=64,
                    label="Width (px)"
                )
                height_slider = gr.Slider(
                    minimum=256,
                    maximum=2048,
                    value=512,
                    step=64,
                    label="Height (px)"
                )
                generate_btn = gr.Button("🎨 Generate Image", variant="primary", size="lg")
                
            with gr.Column(scale=1):
                # Output
                output_image = gr.Image(label="Generated Image", height=512)
                status_text = gr.Textbox(label="Status", interactive=False)
        
        # System info
        with gr.Accordion("System Information", open=False):
            info_text = f"""
            **GPU Available:** {CUDA_AVAILABLE}
            **GPU Name:** {torch.cuda.get_device_name(0) if CUDA_AVAILABLE else 'N/A'}
            **PyTorch Version:** {torch.__version__}
            **Default Device:** {'GPU' if CUDA_AVAILABLE else 'CPU'}
            """
            gr.Markdown(info_text)
        
        # Connect the generate button
        generate_btn.click(
            fn=generate_image,
            inputs=[prompt_input, device_choice, steps_slider, guidance_slider, width_slider, height_slider],
            outputs=[output_image, status_text]
        )
    
    return interface

if __name__ == "__main__":
    print("🚀 Starting AI Image Generator...")
    print(f"🔍 Python version: {torch.__version__}")
    print(f"🔍 CUDA Available: {CUDA_AVAILABLE}")
    
    # Check cache status at startup
    print("\n📊 Cache Status Check:")
    check_cache_status()
    print()
    
    try:
        # Don't pre-load pipelines - use lazy loading instead
        print("ℹ️ Pipelines will be loaded on first use (lazy loading)")
        
        # Create and launch interface
        print("🔄 Creating Gradio interface...")
        interface = create_interface()
        
        print("🌐 Launching web interface...")
        print("📡 Server will be available at:")
        print("   - Local: http://localhost:7860")
        print("   - Local (IP): http://127.0.0.1:7860")
        print("   - Network: http://0.0.0.0:7860")
        print("💡 If localhost doesn't work, try 127.0.0.1:7860")
        
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,  # Disable share link for better performance
            show_error=True,
            debug=False,  # Disable debug mode for performance
            inbrowser=False,  # Don't auto-open browser
            prevent_thread_lock=False  # Ensure server stays running
        )
        
    except Exception as e:
        print(f"❌ Failed to start application: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
