import sys
sys.path.insert(0, './hy3dshape')
sys.path.insert(0, './hy3dpaint')

from mmgp import offload, profile_type

from PIL import Image
from hy3dshape.rembg import BackgroundRemover
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

try:
    from torchvision_fix import apply_fix
    apply_fix()
except ImportError:
    print("Warning: torchvision_fix module not found, proceeding without compatibility fix")
except Exception as e:
    print(f"Warning: Failed to apply torchvision fix: {e}")

# shape
model_path = 'tencent/Hunyuan3D-2.1'
print("Loading shape-generation pipeline... (GPU) ")
pipeline_shapegen = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)

image_path = 'assets/demo.png'
image = Image.open(image_path).convert("RGBA")
if image.mode == 'RGB':
    rembg = BackgroundRemover()
    image = rembg(image)

mesh = pipeline_shapegen(image=image)[0]
mesh.export('demo.glb')

# paint
max_num_view = 6  # can be 6 to 9
resolution = 512  # can be 768 or 512
conf = Hunyuan3DPaintConfig(max_num_view, resolution)

# Make sure the heavy diffusion pipeline is kept on CPU
conf.device = "cpu"

conf.realesrgan_ckpt_path = "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
conf.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
conf.custom_pipeline = "hy3dpaint/hunyuanpaintpbr/pipeline.py"

print("Instantiating paint pipeline...")
paint_pipeline = Hunyuan3DPaintPipeline(conf)

# Apply mmgp memory-management + quantisation
print("Enabling mmgp off-loading / quantisation...")
try:
    core_pipe = paint_pipeline.models["multiview_model"].pipeline
    offload.profile(core_pipe, profile_type.LowRAM_LowVRAM)
except Exception as e:
    print("[mmgp] Failed to apply off-loading profile ➜ continuing without it.\n", e)

output_mesh_path = 'demo_textured.glb'
print("Generating textured mesh... (this might take a while)")
output_mesh_path = paint_pipeline(
    mesh_path = "demo.glb", 
    image_path = 'assets/demo.png',
    output_mesh_path = output_mesh_path
)

print(f"Textured mesh saved to {output_mesh_path}") 
