import torch
import gradio as gr
import spaces

from optimum.quanto import freeze, qfloat8, quantize

from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from transformers import CLIPTextModel, CLIPTokenizer,T5EncoderModel, T5TokenizerFast

# --- Model and Pipeline Setup ---
dtype = torch.bfloat16

bfl_repo = "black-forest-labs/FLUX.1-dev"
revision = "refs/pr/1"
local_path = "FLUX.1-dev-4bit" # Assuming the quantized models are in this local directory

# It's good practice to have a function to load the models to keep the global namespace clean.
def load_pipeline():
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(bfl_repo, subfolder="scheduler", revision=revision)
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14") # Removed torch_dtype for tokenizer
    text_encoder_2 = torch.load(local_path + '/text_encoder_2.pt')
    tokenizer_2 = T5TokenizerFast.from_pretrained(bfl_repo, subfolder="tokenizer_2", revision=revision) # Removed torch_dtype for tokenizer
    vae = AutoencoderKL.from_pretrained(bfl_repo, subfolder="vae", torch_dtype=dtype, revision=revision)
    transformer = torch.load(local_path + '/transformer.pt')

    pipe = FluxPipeline(
        scheduler=scheduler,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        text_encoder_2=None,
        tokenizer_2=tokenizer_2,
        vae=vae,
        transformer=None,
    )
    pipe.text_encoder_2 = text_encoder_2
    pipe.transformer = transformer
    pipe.to('cuda')
    return pipe

# Load the pipeline
pipe = load_pipeline()
print('Pipeline loaded successfully.')

@spaces.GPU
# --- Gradio Inference Function ---
def generate_image(prompt, guidance_scale, num_inference_steps, seed):
    """
    Function to generate an image based on the prompt and other parameters.
    This function will be used by the Gradio interface.
    """
    generator = torch.Generator("cuda").manual_seed(int(seed))
    image = pipe(
        prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=int(num_inference_steps),
        max_sequence_length=256,
        generator=generator
    ).images[0]
    return image

# --- Gradio Interface ---
with gr.Blocks(theme="bethecloud/storj_theme") as demo:
    gr.Markdown("# FLUX.1 Image Generation(4bit)")
    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(label="Prompt", value="a cute apple smiling")
            guidance_scale_slider = gr.Slider(minimum=0.0, maximum=10.0, step=0.1, value=0.0, label="Guidance Scale")
            steps_slider = gr.Slider(minimum=1, maximum=50, step=1, value=4, label="Number of Inference Steps")
            seed_input = gr.Number(label="Seed", value=12345)
            generate_button = gr.Button("Generate Image")
        with gr.Column():
            output_image = gr.Image(label="Generated Image")

    generate_button.click(
        fn=generate_image,
        inputs=[prompt_input, guidance_scale_slider, steps_slider, seed_input],
        outputs=output_image
    )

# --- Launch the Gradio App ---
if __name__ == "__main__":
    demo.launch()