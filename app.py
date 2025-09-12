import spaces
import json
import math
import os
import traceback
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple
import re
import time
from threading import Thread
from io import BytesIO
import uuid
import tempfile

import gradio as gr
import requests
import torch
from PIL import Image
import fitz
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2VLImageProcessor

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Image as RLImage, Paragraph, Spacer
from reportlab.lib.units import inch

# --- Constants and Model Setup ---
MAX_INPUT_TOKEN_LENGTH = 4096
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.__version__ =", torch.__version__)
print("torch.version.cuda =", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("current device:", torch.cuda.current_device())
    print("device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

print("Using device:", device)


# --- Model Loading: tencent/POINTS-Reader ---
MODEL_PATH = 'tencent/POINTS-Reader'

print(f"Loading model: {MODEL_PATH}")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map='auto'
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
image_processor = Qwen2VLImageProcessor.from_pretrained(MODEL_PATH)
print("Model loaded successfully.")


# --- PDF Generation and Preview Utility Function ---
def generate_and_preview_pdf(image: Image.Image, text_content: str, font_size: int, line_spacing: float, alignment: str, image_size: str):
    """
    Generates a PDF, saves it, and then creates image previews of its pages.
    Returns the path to the PDF and a list of paths to the preview images.
    """
    if image is None or not text_content or not text_content.strip():
        raise gr.Error("Cannot generate PDF. Image or text content is missing.")

    # --- 1. Generate the PDF ---
    temp_dir = tempfile.gettempdir()
    pdf_filename = os.path.join(temp_dir, f"output_{uuid.uuid4()}.pdf")
    doc = SimpleDocTemplate(
        pdf_filename,
        pagesize=A4,
        rightMargin=inch, leftMargin=inch,
        topMargin=inch, bottomMargin=inch
    )
    styles = getSampleStyleSheet()
    style_normal = styles["Normal"]
    style_normal.fontSize = int(font_size)
    style_normal.leading = int(font_size) * line_spacing
    style_normal.alignment = {"Left": 0, "Center": 1, "Right": 2, "Justified": 4}[alignment]

    story = []

    img_buffer = BytesIO()
    image.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    
    page_width, _ = A4
    available_width = page_width - 2 * inch
    image_widths = {
        "Small": available_width * 0.3,
        "Medium": available_width * 0.6,
        "Large": available_width * 0.9,
    }
    img_width = image_widths[image_size]
    img = RLImage(img_buffer, width=img_width, height=image.height * (img_width / image.width))
    story.append(img)
    story.append(Spacer(1, 12))

    cleaned_text = re.sub(r'#+\s*', '', text_content).replace("*", "")
    text_paragraphs = cleaned_text.split('\n')
    
    for para in text_paragraphs:
        if para.strip():
            story.append(Paragraph(para, style_normal))

    doc.build(story)

    # --- 2. Render PDF pages as images for preview ---
    preview_images = []
    try:
        pdf_doc = fitz.open(pdf_filename)
        for page_num in range(len(pdf_doc)):
            page = pdf_doc.load_page(page_num)
            pix = page.get_pixmap(dpi=150)
            preview_img_path = os.path.join(temp_dir, f"preview_{uuid.uuid4()}_p{page_num}.png")
            pix.save(preview_img_path)
            preview_images.append(preview_img_path)
        pdf_doc.close()
    except Exception as e:
        print(f"Error generating PDF preview: {e}")
        
    return pdf_filename, preview_images


# --- Core Application Logic ---
@spaces.GPU
def process_document_stream(
    image: Image.Image, 
    prompt_input: str,
    image_scale_factor: float, # New parameter for image scaling
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float
):
    """
    Main function that handles model inference using tencent/POINTS-Reader.
    """
    if image is None:
        yield "Please upload an image.", ""
        return
    if not prompt_input or not prompt_input.strip():
        yield "Please enter a prompt.", ""
        return

    # --- IMPLEMENTATION: Image Scaling based on user input ---
    if image_scale_factor > 1.0:
        try:
            original_width, original_height = image.size
            new_width = int(original_width * image_scale_factor)
            new_height = int(original_height * image_scale_factor)
            print(f"Scaling image from {image.size} to ({new_width}, {new_height}) with factor {image_scale_factor}.")
            # Use a high-quality resampling filter for better results
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        except Exception as e:
            print(f"Error during image scaling: {e}")
            # Continue with the original image if scaling fails
            pass
    # --- END IMPLEMENTATION ---

    temp_image_path = None
    try:
        # --- FIX: Save the PIL Image to a temporary file ---
        # The model expects a file path, not a PIL object.
        temp_dir = tempfile.gettempdir()
        temp_image_path = os.path.join(temp_dir, f"temp_image_{uuid.uuid4()}.png")
        image.save(temp_image_path)
        
        # Prepare content for the model using the temporary file path
        content = [
            dict(type='image', image=temp_image_path),
            dict(type='text', text=prompt_input)
        ]
        messages = [
            {
                'role': 'user',
                'content': content
            }
        ]
        
        # Prepare generation configuration from UI inputs
        generation_config = {
            'max_new_tokens': max_new_tokens,
            'repetition_penalty': repetition_penalty,
            'temperature': temperature,
            'top_p': top_p,
            'top_k': top_k,
            'do_sample': True if temperature > 0 else False
        }

        # Run inference
        response = model.chat(
            messages,
            tokenizer,
            image_processor,
            generation_config
        )
        # Yield the full response at once
        yield response, response

    except Exception as e:
        traceback.print_exc()
        yield f"An error occurred during processing: {str(e)}", ""
    finally:
        # --- Clean up the temporary image file ---
        if temp_image_path and os.path.exists(temp_image_path):
            os.remove(temp_image_path)


# --- Gradio UI Definition ---
def create_gradio_interface():
    """Builds and returns the Gradio web interface."""
    css = """
    .main-container { max-width: 1400px; margin: 0 auto; }
    .process-button { border: none !important; color: white !important; font-weight: bold !important; background-color: blue !important;}
    .process-button:hover { background-color: darkblue !important; transform: translateY(-2px) !important; box-shadow: 0 4px 8px rgba(0,0,0,0.2) !important; }
    #gallery { min-height: 400px; }
    """
    with gr.Blocks(theme="bethecloud/storj_theme", css=css) as demo:
        gr.HTML(f"""
        <div class="title" style="text-align: center">
            <h1>Document Conversion with POINTS Reader 📖</h1>
            <p style="font-size: 1.1em; color: #6b7280; margin-bottom: 0.6em;">
                Using tencent/POINTS-Reader Multimodal for Image Content Extraction
            </p>
        </div>
        """)

        with gr.Row():
            # Left Column (Inputs)
            with gr.Column(scale=1):
                gr.Textbox(
                    label="Model in Use ⚡",
                    value="tencent/POINTS-Reader",
                    interactive=False
                )
                prompt_input = gr.Textbox(
                    label="Query Input",
                    placeholder="✦︎ Enter the prompt",
                    value="Perform OCR on the image precisely.",
                )
                image_input = gr.Image(label="Upload Image", type="pil", sources=['upload'])
                
                with gr.Accordion("Advanced Settings", open=False):
                    # --- NEW UI ELEMENT: Image Scaling Slider ---
                    image_scale_factor = gr.Slider(
                        minimum=1.0, 
                        maximum=3.0, 
                        value=1.0, 
                        step=0.1, 
                        label="Image Upscale Factor",
                        info="Increases image size before processing. Can improve OCR on small text. Default: 1.0 (no change)."
                    )
                    # --- END NEW UI ELEMENT ---
                    max_new_tokens = gr.Slider(minimum=512, maximum=8192, value=2048, step=256, label="Max New Tokens")
                    temperature = gr.Slider(label="Temperature", minimum=0.1, maximum=1.0, step=0.05, value=0.7)
                    top_p = gr.Slider(label="Top-p (nucleus sampling)", minimum=0.05, maximum=1.0, step=0.05, value=0.8)
                    top_k = gr.Slider(label="Top-k", minimum=1, maximum=100, step=1, value=20)
                    repetition_penalty = gr.Slider(label="Repetition penalty", minimum=1.0, maximum=2.0, step=0.05, value=1.05)
                    
                    gr.Markdown("### PDF Export Settings")
                    font_size = gr.Dropdown(choices=["8", "10", "12", "14", "16", "18"], value="12", label="Font Size")
                    line_spacing = gr.Dropdown(choices=[1.0, 1.15, 1.5, 2.0], value=1.15, label="Line Spacing")
                    alignment = gr.Dropdown(choices=["Left", "Center", "Right", "Justified"], value="Justified", label="Text Alignment")
                    image_size = gr.Dropdown(choices=["Small", "Medium", "Large"], value="Medium", label="Image Size in PDF")

                process_btn = gr.Button("🚀 Process Image", variant="primary", elem_classes=["process-button"], size="lg")
                clear_btn = gr.Button("🗑️ Clear All", variant="secondary")

            # Right Column (Outputs)
            with gr.Column(scale=2):
                with gr.Tabs() as tabs:
                    with gr.Tab("📝 Extracted Content"):
                        raw_output_stream = gr.Textbox(label="Raw Model Output (max T ≤ 120s)", interactive=False, lines=15, show_copy_button=True)
                        with gr.Row():
                            examples = gr.Examples(
                                examples=["examples/1.jpeg", 
                                          "examples/2.jpeg", 
                                          "examples/3.jpeg",
                                          "examples/4.jpeg", 
                                          "examples/5.jpeg"],
                                inputs=image_input, label="Examples"
                            )
                        gr.Markdown("[Report-Bug💻](https://huggingface.co/spaces/prithivMLmods/POINTS-Reader-OCR/discussions) | [prithivMLmods🤗](https://huggingface.co/prithivMLmods)")
                    
                    with gr.Tab("📰 README.md"):
                        with gr.Accordion("(Result.md)", open=True): 
                            # --- FIX: Added latex_delimiters to enable LaTeX rendering ---
                            markdown_output = gr.Markdown(latex_delimiters=[
                                {"left": "$$", "right": "$$", "display": True},
                                {"left": "$", "right": "$", "display": False}
                            ])

                    with gr.Tab("📋 PDF Preview"):
                        generate_pdf_btn = gr.Button("📄 Generate PDF & Render", variant="primary")
                        pdf_output_file = gr.File(label="Download Generated PDF", interactive=False)
                        pdf_preview_gallery = gr.Gallery(label="PDF Page Preview", show_label=True, elem_id="gallery", columns=2, object_fit="contain", height="auto")

        # Event Handlers
        def clear_all_outputs():
            return None, "", "Raw output will appear here.", "", None, None

        process_btn.click(
            fn=process_document_stream,
            # --- UPDATE: Add the new slider to the inputs list ---
            inputs=[image_input, prompt_input, image_scale_factor, max_new_tokens, temperature, top_p, top_k, repetition_penalty],
            outputs=[raw_output_stream, markdown_output]
        )
        
        generate_pdf_btn.click(
            fn=generate_and_preview_pdf,
            inputs=[image_input, raw_output_stream, font_size, line_spacing, alignment, image_size],
            outputs=[pdf_output_file, pdf_preview_gallery]
        )

        clear_btn.click(
            clear_all_outputs,
            outputs=[image_input, prompt_input, raw_output_stream, markdown_output, pdf_output_file, pdf_preview_gallery]
        )
    return demo

if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.queue(max_size=50).launch(share=True, show_error=True)
