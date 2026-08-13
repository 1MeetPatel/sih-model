import os
import cv2
import torch
import gradio as gr
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url

from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

def process_image(input_img, model_name, outscale, denoise_strength, face_enhance, tile, fp32):
    if input_img is None:
        return None
    
    model_name = model_name.split('.')[0]
    if model_name == 'RealESRGAN_x4plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
    elif model_name == 'RealESRNet_x4plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth']
    elif model_name == 'RealESRGAN_x4plus_anime_6B':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth']
    elif model_name == 'RealESRGAN_x2plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']
    elif model_name == 'realesr-animevideov3':
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth']
    elif model_name == 'realesr-general-x4v3':
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        netscale = 4
        file_url = [
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
        ]
    else:
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']

    model_path = os.path.join('weights', model_name + '.pth')
    if not os.path.isfile(model_path):
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        for url in file_url:
            model_path = load_file_from_url(
                url=url, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)

    dni_weight = None
    if model_name == 'realesr-general-x4v3' and denoise_strength != 1:
        wdn_model_path = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
        model_path = [model_path, wdn_model_path]
        dni_weight = [denoise_strength, 1 - denoise_strength]

    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=tile,
        tile_pad=10,
        pre_pad=0,
        half=not fp32,
        gpu_id=0 if torch.cuda.is_available() else None)

    if face_enhance:
        from gfpgan import GFPGANer
        face_enhancer = GFPGANer(
            model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
            upscale=outscale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=upsampler)

    img_bgr = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)

    try:
        if face_enhance:
            _, _, output = face_enhancer.enhance(img_bgr, has_aligned=False, only_center_face=False, paste_back=True)
        else:
            output, _ = upsampler.enhance(img_bgr, outscale=outscale)
    except Exception as error:
        print('Error:', error)
        return None

    output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    return output_rgb

def create_ui():
    description = """
    Upload any low-resolution image to restore details, upscale resolution (2x, 4x, etc.), and optionally apply face enhancement using GFPGAN.
    """
    
    with gr.Blocks() as app:
        gr.Markdown(description)

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="numpy", label="Input Image")
                model_name = gr.Dropdown(
                    choices=[
                        'RealESRGAN_x4plus',
                        'RealESRNet_x4plus',
                        'RealESRGAN_x4plus_anime_6B',
                        'RealESRGAN_x2plus',
                        'realesr-animevideov3',
                        'realesr-general-x4v3'
                    ],
                    value='RealESRGAN_x4plus',
                    label="Model"
                )
                outscale = gr.Slider(minimum=1, maximum=8, value=4, step=0.5, label="Outscale (Upsample Ratio)")
                denoise_strength = gr.Slider(minimum=0, maximum=1, value=0.5, step=0.1, label="Denoise Strength (for general-x4v3)")
                face_enhance = gr.Checkbox(label="Face Enhancement (GFPGAN)", value=False)
                tile = gr.Slider(minimum=0, maximum=512, value=0, step=32, label="Tile Size (0 for no tile, use if low GPU memory)")
                fp32 = gr.Checkbox(label="Use FP32 precision (Default FP16 half precision)", value=False)
                submit_btn = gr.Button("🚀 Enhance & Upscale Image", variant="primary")
            
            with gr.Column():
                output_image = gr.Image(label="Restored & Upscaled Result")
        
        sample_images = [
            ['inputs/0014.jpg', 'RealESRGAN_x4plus', 4, 0.5, False, 0, False],
            ['inputs/0030.jpg', 'RealESRGAN_x4plus', 4, 0.5, False, 0, False],
            ['inputs/00003.png', 'RealESRGAN_x4plus', 4, 0.5, False, 0, False],
            ['inputs/00017_gray.png', 'RealESRGAN_x4plus', 4, 0.5, False, 0, False],
        ]
        gr.Examples(
            examples=sample_images,
            inputs=[input_image, model_name, outscale, denoise_strength, face_enhance, tile, fp32],
            outputs=output_image,
            fn=process_image,
            cache_examples=False
        )

        submit_btn.click(
            fn=process_image,
            inputs=[input_image, model_name, outscale, denoise_strength, face_enhance, tile, fp32],
            outputs=output_image
        )

    return app

if __name__ == '__main__':
    app = create_ui()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
