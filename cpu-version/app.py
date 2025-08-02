"""
Simple AI Image Generator
GPU (default/preferred) and CPU modes available
"""

import gradio as gr
import torch
from PIL import Image
import time

# Import AI libraries
try:
    from transformers import CLIPTextModel, CLIPTokenizer
    from diffusers import StableDiffusionPipeline
    print("✅ All required libraries imported successfully")
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

# Global pipeline variables
gpu_pipeline = None
cpu_pipeline = None

def load_gpu_pipeline():
    """Load Stable Diffusion pipeline for GPU"""
    global gpu_pipeline
    if gpu_pipeline is None and CUDA_AVAILABLE:
        print("🔄 Loading GPU pipeline...")
        try:
            gpu_pipeline = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16,
                safety_checker=None,
                requires_safety_checker=False
            )
            gpu_pipeline = gpu_pipeline.to("cuda")
            gpu_pipeline.enable_attention_slicing()
            print("✅ GPU pipeline loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load GPU pipeline: {e}")
            import traceback
            traceback.print_exc()
            return False
    return True

def load_cpu_pipeline():
    """Load Stable Diffusion pipeline for CPU"""
    global cpu_pipeline
    if cpu_pipeline is None:
        print("🔄 Loading CPU pipeline...")
        try:
            cpu_pipeline = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            cpu_pipeline = cpu_pipeline.to("cpu")
            print("✅ CPU pipeline loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load CPU pipeline: {e}")
            import traceback
            traceback.print_exc()
            return False
    return True

def generate_image(prompt, device_choice, num_inference_steps, guidance_scale, width, height):
    """Generate image using selected device"""
    
    if not prompt.strip():
        return None, "❌ Please enter a prompt"
    
    start_time = time.time()
    
    try:
        if device_choice == "GPU" and CUDA_AVAILABLE:
            if not load_gpu_pipeline():
                return None, "❌ Failed to load GPU pipeline"
            
            print(f"🎮 Generating with GPU: {prompt}")
            image = gpu_pipeline(
                prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width
            ).images[0]
            device_used = "GPU"
            
        else:
            if not load_cpu_pipeline():
                return None, "❌ Failed to load CPU pipeline"
            
            print(f"💻 Generating with CPU: {prompt}")
            image = cpu_pipeline(
                prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width
            ).images[0]
            device_used = "CPU"
        
        generation_time = time.time() - start_time
        status = f"✅ Generated using {device_used} in {generation_time:.1f}s"
        
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
    
    try:
        # Pre-load the default pipeline
        if CUDA_AVAILABLE:
            print("🔄 Pre-loading GPU pipeline...")
            load_gpu_pipeline()
        else:
            print("🔄 Pre-loading CPU pipeline...")
            load_cpu_pipeline()
        
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
            share=True,  # Enable share link for external access
            show_error=True,
            debug=True,  # Enable debug mode
            inbrowser=False,  # Don't auto-open browser
            prevent_thread_lock=False  # Ensure server stays running
        )
        
    except Exception as e:
        print(f"❌ Failed to start application: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
