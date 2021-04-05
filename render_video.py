# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import argparse
import os
from pathlib import Path
import time
import numpy as np
import scipy.ndimage

import pickle
import dnnlib
import torch
import legacy

#----------------------------------------------------------------------------
# main function for command line call

def main():

    starttime = int(time.time())
    parser = argparse.ArgumentParser(
        description='Render from StyleGAN2 saved models (pkl files).',
        #epilog=_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--network_pkl', help='The pkl file to render from (the model checkpoint).', default=None, metavar='MODEL.pkl', required=True)
    parser.add_argument('--grid_x', help='Number of images to render horizontally (each frame will have rows of X images, default: 1).', default=1, metavar='X', type=int)
    parser.add_argument('--grid_y', help='Number of images to render vertically (each frame will have cols of Y images, default: 1).', default=1, metavar='Y', type=int)
    parser.add_argument('--image_zoom', help='Zoom on the output image (seems like just more video pixels, but no true upscaling)', default=1, type=float)
    parser.add_argument('--duration_sec', help='Length of video to render in seconds.', default=30.0, type=float)
    parser.add_argument('--mp4_fps', help='Frames per second for video rendering', default=30, type=float)
    parser.add_argument('--smoothing_sec', help='Gaussian kernel size in seconds to blend video frames (higher value = less change, lower value = more erratic, default: 1.0)', default=1.0, type=float)
    parser.add_argument('--truncation_psi', help='Truncation parameter (1 = normal, lower values overfit to look more like originals, higher values underfit to be more abstract, recommendation: 0.5-2)', default=1, type=float)
    parser.add_argument('--noise_mode', help='Either "none", "const", or "random". Can add noise to vary rendered images.', default='none', type=str)
    parser.add_argument('--filename', help='Filename for rendering output, defaults to pkl filename', default=None)
    parser.add_argument('--mp4_codec', help='Video codec to use with moviepy (i.e. libx264, libx265, mpeg4)', default='libx264')
    parser.add_argument('--mp4_bitrate', help='Bitrate to use with moviepy (i.e. 16M)', default='16M')
    parser.add_argument('--random_seed', help='Seed to initialize the latent generation.', default=starttime, type=int)
    parser.add_argument('--outdir', help='Where to save the output video', type=str, metavar='DIR', default='./videos')

    args = parser.parse_args()

    generate_interpolation_video(
        network_pkl=args.network_pkl, 
        grid_size=[args.grid_x, args.grid_y], 
        image_zoom=args.image_zoom, 
        duration_sec=args.duration_sec, 
        smoothing_sec=args.smoothing_sec, 
        truncation_psi=args.truncation_psi,
        noise_mode=args.noise_mode,
        filename=args.filename, 
        mp4_fps=args.mp4_fps, 
        mp4_codec=args.mp4_codec, 
        mp4_bitrate=args.mp4_bitrate,
        random_seed=args.random_seed, 
        outdir=args.outdir
    )

#----------------------------------------------------------------------------
# Helper functions that have been dropped from pgan to sgan

def random_latents(num_latents, G, random_state=None):
    if random_state is not None:
        return random_state.randn(num_latents, *G.input_shape[1:]).astype(np.float32)
    else:
        return np.random.randn(num_latents, *G.input_shape[1:]).astype(np.float32)

def load_pkl(network_pkl):
    with open(network_pkl, 'rb') as file:
        return pickle.load(file, encoding='latin1')

def get_id_string_for_network_pkl(network_pkl):
    filename = network_pkl.replace('.pkl', '').replace('\\', '/').split('/')[-1]
    return filename

# and from sgan2 to sgan2-ada-pytorch

def create_image_grid(images, grid_size=None):
    assert images.ndim == 3 or images.ndim == 4
    num, img_w, img_h = images.shape[0], images.shape[-1], images.shape[-2]

    if grid_size is not None:
        grid_w, grid_h = tuple(grid_size)
    else:
        grid_w = max(int(np.ceil(np.sqrt(num))), 1)
        grid_h = max((num - 1) // grid_w + 1, 1)

    grid = np.zeros(list(images.shape[1:-2]) + [grid_h * img_h, grid_w * img_w], dtype=images.dtype)
    for idx in range(num):
        x = (idx % grid_w) * img_w
        y = (idx // grid_w) * img_h
        grid[..., y : y + img_h, x : x + img_w] = images[idx]
    return grid.transpose(1, 2, 0)

#----------------------------------------------------------------------------
# Generate MP4 video of random interpolations using a previously trained network.
# To run, uncomment the appropriate line in config.py and launch train.py.


def generate_interpolation_video(network_pkl = None, grid_size=[1,1], png_sequence=False, image_zoom=1, duration_sec=60.0, smoothing_sec=1.0, truncation_psi=1, noise_mode=False, filename=None, mp4_fps=30, mp4_codec='libx264', mp4_bitrate='16M', random_seed=1000, outdir='./videos'):
    
    if network_pkl == None:
        print('ERROR: Please enter pkl path.')
        sys.exit(1)
    num_frames = int(np.rint(duration_sec * mp4_fps))
    random_state = np.random.RandomState(random_seed)
    if filename is None:
        filename = get_id_string_for_network_pkl(network_pkl) + '-seed-' + str(random_seed)

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    print('Generating latent vectors...')
    shape = [num_frames, np.prod(grid_size)] + [G.z_dim] # [frame, image, channel, component]
    print(shape)
    all_latents = random_state.randn(*shape).astype(np.float32)
    all_latents = scipy.ndimage.gaussian_filter(all_latents, [smoothing_sec * mp4_fps] + [0] * (len(shape)-1), mode='wrap')
    all_latents /= np.sqrt(np.mean(np.square(all_latents)))

    
    print("Rendering...\ntruncation_psi =", truncation_psi, ", noise_mode =", noise_mode)
    os.makedirs(outdir, exist_ok=True)


    ###
    ### this is the moviepy implementation of rendering
    ### it has a nice progress bar
    ### there is an imageio implementation commented out below as well
    ###

    # Frame generation func for moviepy.
    # def make_frame(t):
    #     frame_idx = int(np.clip(np.round(t * mp4_fps), 0, num_frames - 1))
    #     z = torch.from_numpy(all_latents[frame_idx]).to(device)
    #     label = np.zeros([z.shape[0], 0], np.float32)
    #     images = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
    #     images = (images * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
    #     grid = create_image_grid(images, grid_size)
    #     if image_zoom > 1:
    #         grid = scipy.ndimage.zoom(grid, [image_zoom, image_zoom, 1], order=0)
    #     if grid.shape[2] == 1:
    #         grid = grid.repeat(3, 2) # grayscale => RGB
    #     return grid

    # # Generate video.
    # import moviepy.editor # pip install moviepy
    # moviepy.editor.VideoClip(make_frame, duration=duration_sec).write_videofile(os.path.join(outdir, filename + ".mp4"), fps=mp4_fps, codec=mp4_codec, bitrate=mp4_bitrate)


    ###
    ### this is an alternative imageio implementation of rendering
    ### I like moviepy more cause it has a nice progress bar
    ### not sure which one is more "performant", feel free to experiment
    ###

    import imageio # pip install imageio
    video = imageio.get_writer(f'{outdir}/seed{random_seed:04d}.mp4', mode='I', fps=mp4_fps, codec=mp4_codec, bitrate=mp4_bitrate)
    for frame_idx in range(num_frames):
        z = torch.from_numpy(all_latents[frame_idx]).to(device)
        label = torch.zeros([1, G.c_dim], device=device)
        img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
        grid = create_image_grid(img, grid_size)
        if image_zoom > 1:
            grid = scipy.ndimage.zoom(grid, [image_zoom, image_zoom, 1], order=0)
        video.append_data(grid)
    video.close()

if __name__ == "__main__":
    main()