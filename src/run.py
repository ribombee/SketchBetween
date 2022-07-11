from pathlib import Path
import argparse
from tqdm import tqdm
from tensorflow_addons.optimizers import lookahead
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import tensorflow as tf
import cv2 as cv
from generator import VQVAEAnimatedGenerator
from plotting import *
import numpy as np

def load_our_model(model_path: Path):

    model = tf.keras.models.load_model(model_path,
                                       compile=False
                                       #custom_objects= {"Addons>Lookahead": lookahead}
                                       )
    print('--------MODEL LOADED---------')
    print(model.summary())
    return model

def run_model_on_data(model_path: Path, data_path: Path, result_path: Path, save_loc: Path = None, include_keyframes=False, do_evaluate=True):

    # Load model.
    model = load_our_model(model_path)
    do_sketches = True
    do_final_keyframe = False

    # Create generator that can fetch the data.
    gen = VQVAEAnimatedGenerator(data_path,
                                 batch_size=1,
                                 shuffle=False,
                                 animation_len=5,
                                 augment=False,
                                 do_sketches=do_sketches,
                                 include_final_keyframe=do_final_keyframe)

    # Run every datapoint through the model and save the outputs to a folder.
    # For every index (starting frame), we create a folder that contains the output frames for that run.

    ssim_list = []
    psnr_list = []

    frame_range = range(5)

    sketch_path = result_path / "sketches"
    recreation_path = result_path / "recreations"
    if not sketch_path.exists():
        sketch_path.mkdir()
    if not recreation_path.exists():
        recreation_path.mkdir()

    for gen_idx in tqdm(range(gen.__len__())):

        x, y_true = gen.__getitem__(gen_idx)

        if do_evaluate:
            y_pred = model.predict(x)

            for idx in frame_range:

                frame_ssim = ssim(y_true[0][idx], y_pred[0][idx], channel_axis=2)
                frame_psnr = psnr(y_true[0][idx], y_pred[0][idx])

                #Serialize frames!
                cv.imwrite(str(sketch_path / f'{gen_idx}_{idx}.png'), x[0][idx]*255)
                cv.imwrite(str(recreation_path / f'{gen_idx}_{idx}.png'), cv.cvtColor(y_pred[0][idx]*255, cv.COLOR_RGB2BGR))

                if idx not in range(1,4) and not include_keyframes:
                    continue

                ssim_list.append(frame_ssim)
                psnr_list.append(frame_psnr)
        else:
            y_pred = model.predict(y_true) #In this case the y_true already contains sketches

            for idx in frame_range:
                # Serialize frames!
                cv.imwrite(str(sketch_path / f'{gen_idx}_{idx}.png'),  y_true[0][idx] * 255)
                cv.imwrite(str(recreation_path / f'{gen_idx}_{idx}.png'),
                           cv.cvtColor(y_pred[0][idx] * 255, cv.COLOR_RGB2BGR))

        if save_loc is not None:
            subplot_animation(input, y_true, y_pred, save_loc=str(save_loc / str(idx)), animation_len=5)

    if do_evaluate:
        print(f'AVERAGE SSIM IS {np.array(ssim_list).mean()}')
        print(f'AVERAGE PSNR IS {np.array(psnr_list).mean()}')


def __parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_loc", type=str)
    parser.add_argument("--data_loc", type=str)
    parser.add_argument("--output_loc", type=str)

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--evaluate", action="store_true")
    group.add_argument("--animate", action="store_true")

    args = parser.parse_args()

    return args.model_loc, args.data_loc, args.output_loc, args.evaluate, args.animate

if __name__ == "__main__":

    model_loc, data_loc, output_loc, do_evaluate, do_animate = __parse_args()
    model_path, data_path, output_path = Path(model_loc), Path(data_loc), Path(output_loc)

    run_model_on_data(model_path, data_path, output_path, do_evaluate=do_evaluate)

    pass