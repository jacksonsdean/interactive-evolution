# @title Licensed under the MIT License

# Copyright (c) 2021 Katherine Crowson

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import os
import copy
import time
import moviepy.editor as mpe
import imageio
# from zmq import device
def add_audio_to_video(video_name, audio, fps):
    audio = mpe.AudioFileClip(audio)
    video = mpe.VideoFileClip(video_name)
    video = mpe.CompositeVideoClip([video])
    final = video.set_audio(audio)
    video_name = video_name.replace("tmp/", "")
    final.write_videofile("videos/edited"+"_"+video_name.replace(".gif", ".mp4"), fps=fps)
    final.close()
    video.close()
    audio.close()


def save_video(images=None, name = None, fps = 24, audio=None, show=False, dir =None):
    write_fps = fps
    if name is None:
        name = f'videos/gan_{time.time()}.mp4'
    
    writer = imageio.get_writer(name, fps=write_fps)
    if dir is not None:
        for file in sorted(os.listdir(dir)):
            if not file.endswith(".png") and not file.endswith(".jpg"):
                continue
            im = imageio.imread(f"{dir}/{file}")
            writer.append_data(im)
    elif images is not None:
        for im in images:
            writer.append_data(im)
    else:
        raise(Exception("must specify either images or dir"))

    writer.close()
    time.sleep(1)
    if audio is not None:
        add_audio_to_video(name, audio, fps)
        time.sleep(1)
        # delete name because we made a new version with audio
        if os.path.exists(name):
            os.remove(name)
    # if show:
        # display(Video(name))
    
imagenet_1024 = False #@param {type:"boolean"}
imagenet_16384 = True #@param {type:"boolean"}
gumbel_8192 = False #@param {type:"boolean"}
coco = False #@param {type:"boolean"}
faceshq = False #@param {type:"boolean"}
wikiart_1024 = False #@param {type:"boolean"}
wikiart_16384 = False #@param {type:"boolean"}
sflckr = False #@param {type:"boolean"}
ade20k = False #@param {type:"boolean"}
ffhq = False #@param {type:"boolean"}
celebahq = False #@param {type:"boolean"}

# %%
 
import argparse
import math
from pathlib import Path
import sys
 
sys.path.append('./taming-transformers')
from IPython import display
from base64 import b64encode
from omegaconf import OmegaConf
from PIL import Image
from taming.models import cond_transformer, vqgan
import torch
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm.notebook import tqdm
 
from CLIP import clip
import kornia.augmentation as K
import numpy as np
import imageio
from PIL import ImageFile, Image
# from imgtag import ImgTag    # metadatos 
# from libxmp import *         # metadatos
# import libxmp                # metadatos
from stegano import lsb
import json
ImageFile.LOAD_TRUNCATED_IMAGES = True
 
def sinc(x):
    return torch.where(x != 0, torch.sin(math.pi * x) / (math.pi * x), x.new_ones([]))
 
 
def lanczos(x, a):
    cond = torch.logical_and(-a < x, x < a)
    out = torch.where(cond, sinc(x) * sinc(x/a), x.new_zeros([]))
    return out / out.sum()
 
 
def ramp(ratio, width):
    n = math.ceil(width / ratio + 1)
    out = torch.empty([n])
    cur = 0
    for i in range(out.shape[0]):
        out[i] = cur
        cur += ratio
    return torch.cat([-out[1:].flip([0]), out])[1:-1]
 
 
def resample(input, size, align_corners=True):
    n, c, h, w = input.shape
    dh, dw = size
 
    input = input.view([n * c, 1, h, w])
 
    if dh < h:
        kernel_h = lanczos(ramp(dh / h, 2), 2).to(input.device, input.dtype)
        pad_h = (kernel_h.shape[0] - 1) // 2
        input = F.pad(input, (0, 0, pad_h, pad_h), 'reflect')
        input = F.conv2d(input, kernel_h[None, None, :, None])
 
    if dw < w:
        kernel_w = lanczos(ramp(dw / w, 2), 2).to(input.device, input.dtype)
        pad_w = (kernel_w.shape[0] - 1) // 2
        input = F.pad(input, (pad_w, pad_w, 0, 0), 'reflect')
        input = F.conv2d(input, kernel_w[None, None, None, :])
 
    input = input.view([n, c, h, w])
    return F.interpolate(input, size, mode='bicubic', align_corners=align_corners)
 
 
class ReplaceGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_forward, x_backward):
        ctx.shape = x_backward.shape
        return x_forward
 
    @staticmethod
    def backward(ctx, grad_in):
        return None, grad_in.sum_to_size(ctx.shape)
 
 
replace_grad = ReplaceGrad.apply
 
 
class ClampWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min, max):
        ctx.min = min
        ctx.max = max
        ctx.save_for_backward(input)
        return input.clamp(min, max)
 
    @staticmethod
    def backward(ctx, grad_in):
        input, = ctx.saved_tensors
        return grad_in * (grad_in * (input - input.clamp(ctx.min, ctx.max)) >= 0), None, None
 
 
clamp_with_grad = ClampWithGrad.apply
 
 
def vector_quantize(x, codebook):
    d = x.pow(2).sum(dim=-1, keepdim=True) + codebook.pow(2).sum(dim=1) - 2 * x @ codebook.T
    indices = d.argmin(-1)
    x_q = F.one_hot(indices, codebook.shape[0]).to(d.dtype) @ codebook
    return replace_grad(x_q, x)
 
 
class Prompt(nn.Module):
    def __init__(self, embed, weight=1., stop=float('-inf')):
        super().__init__()
        self.register_buffer('embed', embed)
        self.register_buffer('weight', torch.as_tensor(weight))
        self.register_buffer('stop', torch.as_tensor(stop))
 
    def forward(self, input):
        input_normed = F.normalize(input.unsqueeze(1), dim=2)
        embed_normed = F.normalize(self.embed.unsqueeze(0), dim=2)
        dists = input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2)
        dists = dists * self.weight.sign()
        return self.weight.abs() * replace_grad(dists, torch.maximum(dists, self.stop)).mean()
 
 
def parse_prompt(prompt):
    vals = prompt.rsplit(':', 2)
    vals = vals + ['', '1', '-inf'][len(vals):]
    return vals[0], float(vals[1]), float(vals[2])
 
 
class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow
        self.augs = nn.Sequential(
            K.RandomHorizontalFlip(p=0.5),
            # K.RandomSolarize(0.01, 0.01, p=0.7),
            K.RandomSharpness(0.3,p=0.4),
            K.RandomAffine(degrees=30, translate=0.1, p=0.8, padding_mode='border'),
            K.RandomPerspective(0.2,p=0.4),
            K.ColorJitter(hue=0.01, saturation=0.01, p=0.7))
        self.noise_fac = 0.1
 
 
    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
        batch = self.augs(torch.cat(cutouts, dim=0))
        if self.noise_fac:
            facs = batch.new_empty([self.cutn, 1, 1, 1]).uniform_(0, self.noise_fac)
            batch = batch + facs * torch.randn_like(batch)
        return batch
 
 
def load_vqgan_model(config_path, checkpoint_path):
    config = OmegaConf.load(config_path)
    if config.model.target == 'taming.models.vqgan.VQModel':
        model = vqgan.VQModel(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
    elif config.model.target == 'taming.models.cond_transformer.Net2NetTransformer':
        parent_model = cond_transformer.Net2NetTransformer(**config.model.params)
        parent_model.eval().requires_grad_(False)
        parent_model.init_from_ckpt(checkpoint_path)
        model = parent_model.first_stage_model
    elif config.model.target == 'taming.models.vqgan.GumbelVQ':
        model = vqgan.GumbelVQ(**config.model.params)
        print(config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
    else:
        raise ValueError(f'unknown model type: {config.model.target}')
    del model.loss
    return model
 
 
def resize_image(image, out_size):
    ratio = image.size[0] / image.size[1]
    area = min(image.size[0] * image.size[1], out_size[0] * out_size[1])
    size = round((area * ratio)**0.5), round((area / ratio)**0.5)
    return image.resize(size, Image.LANCZOS)

#@title Parámetros

cancel = False

def cancel_gan():
    global cancel
    cancel = True


nombres_modelos={"vqgan_imagenet_f16_16384": 'ImageNet 16384',"vqgan_imagenet_f16_1024":"ImageNet 1024", 
                "wikiart_1024":"WikiArt 1024", "wikiart_16384":"WikiArt 16384", "coco":"COCO-Stuff", "faceshq":"FacesHQ", "sflckr":"S-FLCKR", "ade20k":"ADE20K", "ffhq":"FFHQ", "celebahq":"CelebA-HQ", "gumbel_8192": "Gumbel 8192"}


def run_gan(text, image=None, width=512, height=512, progress_callback=None, done_callback = None, max_iteraciones = 250, seed=-1, custom_save=None, one_train=False, frames=None, intervalo_imagenes=10, modelo="vqgan_imagenet_f16_1024"):
    global cancel
    cancel = False
    textos = text
    # modelo = "vqgan_imagenet_f16_16384" #@param ["vqgan_imagenet_f16_16384", "vqgan_imagenet_f16_1024", "wikiart_1024", "wikiart_16384", "coco", "faceshq", "sflckr", "ade20k", "ffhq", "celebahq", "gumbel_8192"]
    # modelo = "wikiart_16384" #@param ["vqgan_imagenet_f16_16384", "vqgan_imagenet_f16_1024", "wikiart_1024", "wikiart_16384", "coco", "faceshq", "sflckr", "ade20k", "ffhq", "celebahq", "gumbel_8192"]
    imagen_inicial = image #@param {type:"string"}
    imagenes_objetivo = None#@param {type:"string"}
    input_images = ""
    nombre_modelo = nombres_modelos[modelo]     

    if modelo == "gumbel_8192":
        is_gumbel = True
    else:
        is_gumbel = False

    if seed == -1:
        seed = None
    if imagen_inicial == "None":
        imagen_inicial = None
    # elif imagen_inicial and imagen_inicial.lower().startswith("http"):
        # imagen_inicial = download_img(imagen_inicial)

    if imagenes_objetivo == "None" or not imagenes_objetivo:
        imagenes_objetivo = []
    else:
        imagenes_objetivo = imagenes_objetivo.split("|")
        imagenes_objetivo = [image.strip() for image in imagenes_objetivo]

    if imagen_inicial or imagenes_objetivo != []:
        input_images = True

    textos = [frase.strip() for frase in textos.split("|")]
    if textos == ['']:
        textos = []
    
    
    #@title Hacer la ejecución...
    # device = torch.device('cpu')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    global i
    for run in range(1):
        texts = text
        for f in os.listdir("steps"):
            os.remove(os.path.join("steps", f))

        args = argparse.Namespace(
            prompts=texts,
            image_prompts=imagenes_objetivo,
            noise_prompt_seeds=[],
            noise_prompt_weights=[],
            size=[width, height],
            init_image=imagen_inicial,
            init_weight=0.,
            clip_model='ViT-B/32',
            vqgan_config=f'{modelo}.yaml',
            vqgan_checkpoint=f'{modelo}.ckpt',
            step_size=0.1,
            cutn=64,
            cut_pow=1.,
            display_freq=intervalo_imagenes,
            seed=seed,
        )
        print(f"run {run}")
        print('Using device:', device)
        if texts:
            print('Using texts:', texts)
        if imagenes_objetivo:
            print('Using image prompts:', imagenes_objetivo)
        if args.seed is None:
            seed = torch.seed()
        else:
            seed = args.seed
        torch.manual_seed(seed)
        print('Using seed:', seed)
        model = load_vqgan_model(args.vqgan_config, args.vqgan_checkpoint).to(device)
        perceptor = clip.load(args.clip_model, jit=False)[0].eval().requires_grad_(False).to(device)

        cut_size = perceptor.visual.input_resolution
        if is_gumbel:
            e_dim = model.quantize.embedding_dim
        else:
            e_dim = model.quantize.e_dim

        f = 2**(model.decoder.num_resolutions - 1)
        make_cutouts = MakeCutouts(cut_size, args.cutn, cut_pow=args.cut_pow)
        if is_gumbel:
            n_toks = model.quantize.n_embed
        else:
            n_toks = model.quantize.n_e

        toksX, toksY = args.size[0] // f, args.size[1] // f
        sideX, sideY = toksX * f, toksY * f
        if is_gumbel:
            z_min = model.quantize.embed.weight.min(dim=0).values[None, :, None, None]
            z_max = model.quantize.embed.weight.max(dim=0).values[None, :, None, None]
        else:
            z_min = model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
            z_max = model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]

        if args.init_image:
            print(f"Using initial image: {args.init_image}")
            pil_image = Image.open(args.init_image).convert('RGB')
            pil_image = pil_image.resize((sideX, sideY), Image.LANCZOS)
            z, *_ = model.encode(TF.to_tensor(pil_image).to(device).unsqueeze(0) * 2 - 1)
        else:
            one_hot = F.one_hot(torch.randint(n_toks, [toksY * toksX], device=device), n_toks).float()
            if is_gumbel:
                z = one_hot @ model.quantize.embed.weight
            else:
                z = one_hot @ model.quantize.embedding.weight
            z = z.view([-1, toksY, toksX, e_dim]).permute(0, 3, 1, 2)
        z_orig = z.clone()
        z.requires_grad_(True)
        opt = optim.Adam([z], lr=args.step_size)

        normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                        std=[0.26862954, 0.26130258, 0.27577711])

        pMs = []

        for prompt in args.prompts:
            txt, weight, stop = parse_prompt(prompt)
            embed = perceptor.encode_text(clip.tokenize(txt).to(device)).float()
            pMs.append(Prompt(embed, weight, stop).to(device))

        for prompt in args.image_prompts:
            path, weight, stop = parse_prompt(prompt)
            img = resize_image(Image.open(path).convert('RGB'), (sideX, sideY))
            batch = make_cutouts(TF.to_tensor(img).unsqueeze(0).to(device))
            embed = perceptor.encode_image(normalize(batch)).float()
            pMs.append(Prompt(embed, weight, stop).to(device))

        for seed, weight in zip(args.noise_prompt_seeds, args.noise_prompt_weights):
            gen = torch.Generator().manual_seed(seed)
            embed = torch.empty([1, perceptor.visual.output_dim]).normal_(generator=gen)
            pMs.append(Prompt(embed, weight).to(device))

        def synth(z):
            if is_gumbel:
                z_q = vector_quantize(z.movedim(1, 3), model.quantize.embed.weight).movedim(3, 1)
            else:
                z_q = vector_quantize(z.movedim(1, 3), model.quantize.embedding.weight).movedim(3, 1)
            
            return clamp_with_grad(model.decode(z_q).add(1).div(2), 0, 1)

        def add_xmp_data(nombrefichero):
            return # Todo

        def add_stegano_data(filename):
            return
            # data = {
            #     "title": " | ".join(args.prompts) if args.prompts else None,
            #     "notebook": "VQGAN+CLIP",
            #     "i": i,
            #     "model": nombre_modelo,
            #     "seed": str(seed),
            #     "input_images": input_images
            # }
            # lsb.hide(filename, json.dumps(data)).save(filename)

        @torch.no_grad()
        def checkin(i, losses):
            losses_str = ', '.join(f'{loss.item():g}' for loss in losses)
            tqdm.write(f'i: {i}, loss: {sum(losses).item():g}, losses: {losses_str}')
            out = synth(z)
            TF.to_pil_image(out[0].cpu()).save('progress.png')
            if progress_callback is not None:
                progress_callback(i/max_iteraciones, 'progress.png')
            add_stegano_data('progress.png')
            add_xmp_data('progress.png')
            display.display(display.Image('progress.png'))

        def ascend_txt():
            global i
            out = synth(z)
            iii = perceptor.encode_image(normalize(make_cutouts(out))).float()

            result = []

            if args.init_weight:
                result.append(F.mse_loss(z, z_orig) * args.init_weight / 2)

            for prompt in pMs:
                result.append(prompt(iii))
            img = np.array(out.mul(255).clamp(0, 255)[0].cpu().detach().numpy().astype(np.uint8))[:,:,:]
            img = np.transpose(img, (1, 2, 0))
            filename = f"steps/{i:04}.png"
            imageio.imwrite(filename, np.array(img))
            add_stegano_data(filename)
            add_xmp_data(filename)
            return result

        def train(i):
            opt.zero_grad()
            lossAll = ascend_txt()
            if i % args.display_freq == 0:
                checkin(i, lossAll)
            loss = sum(lossAll)
            loss.backward()
            opt.step()
            with torch.no_grad():
                z.copy_(z.maximum(z_min).minimum(z_max))

        i = 0
        try:
            with tqdm() as pbar:
                while True:
                    if cancel:
                        break
                    train(i)
                    if i == max_iteraciones:
                        break
                    i += 1
                    
                    pbar.update()
                    if progress_callback is not None:
                        progress_callback(i, None)
                            
                pbar.close()
        except KeyboardInterrupt:
            pass
        #   save_video(textsos)
        out = synth(z)  
        if custom_save is not None:
            if one_train:
                with tqdm(total=len(frames)) as pbar2:
                    for frame_i, frame in enumerate(frames):
                        with torch.no_grad():
                            current_file = frame
                            for iter_index in range(10):
                                pil_image = pil_image.resize((sideX, sideY), Image.LANCZOS)
                                pil_image = Image.open(current_file).convert('RGB')
                                z, *_ = model.encode(TF.to_tensor(pil_image).to(device).unsqueeze(0) * 2 - 1)
                                out = synth(z)
                                img = np.array(out.mul(255).clamp(0, 255)[0].cpu().detach().numpy().astype(np.uint8))[:,:,:]
                                img = np.transpose(img, (1, 2, 0))
                                current_file = f"steps/{i:04}_iter_{iter_index:04}.png"
                                imageio.imwrite(current_file, np.array(img))
                                z.copy_(z.maximum(z_min).minimum(z_max))
                            out = synth(z)  
                            TF.to_pil_image(out[0].cpu()).save(f"{custom_save.replace('.png', f'_{frame_i:04}.png')}")
                        pbar2.update()
                    pbar2.close()
            else:
                TF.to_pil_image(out[0].cpu()).save(custom_save)
            # todo save steps?
            
            done_callback(custom_save)
        else:
            TF.to_pil_image(out[0].cpu()).save(f'{time.time()}_output_{texts[0]}.png')
            save_video(dir="steps", name=f"{time.time()}_{texts[0]}.mp4")
        #   save_latest()
    


if __name__ == '__main__':
    run_gan("testing", None)



# import copy
# import cv2
# import os
    
    # Opens the Video file
    # cap= cv2.VideoCapture('targets/simpsons.mp4')
    # iter=0
    # for file in os.listdir('frames'):
        # os.remove(os.path.join('frames',file))

    # while(cap.isOpened()):
        # ret, frame = cap.read()
        # if ret == False:
            # break
        # cv2.imwrite(f'frames/{iter:05d}.jpg',frame)
        # iter+=1
        
    # cap.release()
    # cv2.destroyAllWindows()
    # max_iteraciones = 10
    # imagenes_objetivo = []
    # width=1280//4
    # height = 720//4
    # seed = 42
    # for file in os.listdir('tmp'):
    #     os.remove(os.path.join('tmp',file))

    # initial_images = [copy.deepcopy(imagen_inicial)]
    # all_frames = [os.path.join("frames", img) for img in os.listdir('frames')][::2]
    # for frame_num, img in enumerate(os.listdir('frames')[::2]):
        # run_gan(text=["the simpsons in starlight playing jazz"], image=os.path.join("frames", img), seed=seed, custom_save=f"tmp/{frame_num:05d}.png")
    # run_gan(text=["the simpsons in starlight playing jazz"], image=all_frames[len(all_frames)//2], frames=all_frames, seed=seed, one_train=True, custom_save=f"tmp/frame.png")
    # save_video(dir="tmp", name=f"simpsons_in_starlight_{time.time()}.mp4", fps=15, audio="targets/simpsons.wav")

    # %%
    # save_video(dir="tmp", name=f"simpsons_in_starlight_{time.time()}.mp4", fps=15, audio="targets/simpsons.wav")


    # %% [markdown]
    # ## Genera un vídeo con los resultados
    # 
    # Si quieres generar un vídeo con los frames, solo haz click abajo. Puedes modificar el número de FPS, el frame inicial, el último frame, etc.

    # %%
    # save_video()

    # %%
    # @title Ver vídeo en el navegador
    # @markdown Este proceso puede tardar un poco más. Si no quieres esperar, descárgalo ejecutando la celda siguiente en vez de usar esta celda.
    # mp4 = open('video.mp4','rb').read()
    # data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
    # display.HTML("""
    # <video width=400 controls>
        # <source src="%s" type="video/mp4">
    # </video>
    # """ % data_url)

    # %%
    # @title Descargar vídeo
    # from google.colab import files
    # files.download("video.mp4")


