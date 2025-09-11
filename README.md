# POINTS-Reader OCR

A powerful document conversion application powered by the POINTS-Reader vision-language model for end-to-end optical character recognition and document processing.

### Overview

POINTS-Reader is a distillation-free Vision-Language Model that achieves state-of-the-art performance on document conversion tasks. This application provides a user-friendly interface for extracting text from document images with high accuracy and generates formatted PDF outputs.

| ![Image 3](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/8UzhFvAxm8g0bIwCJu7W0.png) |  ![Image 1](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/IfivFLIH8ZsXQOzU3gUiu.png) |  ![Image 2](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/Dn_jg_CCwPpNLZR70KMmW.png) | 
|---|---|---|
| Preview 1 | Preview 2 | Preview 3 |

**Key Features:**
- End-to-end document conversion from images
- Advanced OCR with vision-language understanding
- PDF generation with customizable formatting
- Real-time processing with GPU acceleration
- Interactive web interface built with Gradio

## Model Information

- **Model:** `tencent/POINTS-Reader`
- **Type:** Vision-Language Model
- **Architecture:** Multimodal transformer with Qwen2VL components
- **Performance:** State-of-the-art benchmarks on document understanding tasks

## Installation

### Prerequisites

Ensure you have Python 3.8+ and pip 23.0.0 or higher:

```bash
pip>=23.0.0
```

### Dependencies

Install the required packages:

```bash
git+https://github.com/Dao-AILab/flash-attention.git
git+https://github.com/huggingface/accelerate.git
git+https://github.com/WePOINTS/WePOINTS.git
git+https://github.com/huggingface/peft.git
transformers-stream-generator
transformers==4.55.2
huggingface_hub
albumentations
qwen-vl-utils
pyvips-binary
sentencepiece
opencv-python
docling-core
python-docx
torchvision
safetensors
matplotlib
num2words
reportlab
xformers
requests
pymupdf
hf_xet
spaces
pyvips
pillow
gradio
einops
torch
fpdf
timm
av
```

Or install from requirements.txt:

```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

1. Clone the repository:
```bash
git clone https://github.com/PRITHIVSAKTHIUR/POINTS-Reader-OCR.git
cd POINTS-Reader-OCR
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. Open your browser and navigate to the provided local URL to access the interface.

### Basic Workflow

1. **Upload Image:** Select a document image (JPEG, PNG, etc.)
2. **Enter Prompt:** Use the default OCR prompt or customize for specific tasks
3. **Process:** Click "Process Image" to extract text content
4. **Export:** Generate a formatted PDF with the extracted content

### Advanced Configuration

The application supports various parameters for fine-tuning model behavior:

- **Max New Tokens:** Controls output length (512-8192)
- **Temperature:** Adjusts randomness in generation (0.1-1.0)
- **Top-p:** Nucleus sampling parameter (0.05-1.0)
- **Top-k:** Limits vocabulary during generation (1-100)
- **Repetition Penalty:** Reduces repetitive output (1.0-2.0)

### PDF Generation Options

Customize the output PDF format:

- **Font Size:** 8-18pt text sizing
- **Line Spacing:** 1.0x to 2.0x spacing options
- **Text Alignment:** Left, Center, Right, or Justified
- **Image Size:** Small, Medium, or Large image scaling

## API Reference

### Core Functions

#### `process_document_stream()`
Main inference function that processes document images using the POINTS-Reader model.

**Parameters:**
- `image`: PIL Image object
- `prompt_input`: Text prompt for processing
- `max_new_tokens`: Maximum tokens to generate
- `temperature`: Sampling temperature
- `top_p`: Nucleus sampling threshold
- `top_k`: Top-k sampling parameter
- `repetition_penalty`: Repetition penalty factor

**Returns:**
- Generator yielding processed text content

#### `generate_and_preview_pdf()`
Creates formatted PDF documents from extracted content.

**Parameters:**
- `image`: Source document image
- `text_content`: Extracted text content
- `font_size`: PDF font size
- `line_spacing`: Line spacing multiplier
- `alignment`: Text alignment option
- `image_size`: Image scaling in PDF

**Returns:**
- PDF file path and preview image paths

## Hardware Requirements

- **Recommended:** NVIDIA GPU with CUDA support
- **Minimum:** 8GB RAM, modern CPU
- **Storage:** 5GB for model weights and dependencies

## Performance

The POINTS-Reader model delivers:
- High accuracy OCR on various document types
- Support for complex layouts and formatting
- Multilingual text recognition capabilities
- Fast processing with GPU acceleration

## Examples

The application includes sample document images demonstrating various use cases:
- Financial documents
- Technical papers
- Forms and certificates
- Handwritten notes
- Multi-column layouts

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests to improve the application.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](https://github.com/PRITHIVSAKTHIUR/POINTS-Reader-OCR?tab=Apache-2.0-1-ov-file) file for details.

## Acknowledgments

- **Model:** Tencent POINTS-Reader team for the vision-language model
- **Framework:** Hugging Face Transformers and Gradio communities
- **Libraries:** PyTorch, ReportLab, and other open-source dependencies

## Support

For issues, questions, or feature requests:
- GitHub Issues: [Report bugs or request features](https://huggingface.co/spaces/prithivMLmods/POINTS-Reader-OCR/discussions)
- Developer: [prithivMLmods](https://huggingface.co/prithivMLmods)
