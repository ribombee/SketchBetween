import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tensorflow import keras
from pathlib import Path
import tensorflow as tf
import tensorflow_addons as tfa
from datetime import datetime
import argparse

from generator import VQVAEAnimatedGenerator
from models import *
from losses import *
from plotting import *


# matplotlib.use("TkAgg") # To not use pycharm sciview


class plotting_callback(keras.callbacks.Callback):
    '''
    This is a callback that creates plots at the end of each epoch.
    The plots show an input images
    '''

    def __init__(self, frame_loc, save_loc, animation_len, do_sketches=True, do_second_keyframe=True):
        '''
        :param frame_loc: Must contain animation frames in the form [animation_name]_frame[N]
        :param save_loc: Where the plots will be saved
        :param animation_len: Length of the animation in frames
        :param do_sketches: Whether to include sketches in the input to the model. Used for ablation study.
        :param do_second_keyframe: Whether to include the second keyframe in the input to the model. Used for ablation study.
        '''
        self.gen = VQVAEAnimatedGenerator(dataloc=frame_loc,
                                          batch_size=1,
                                          animation_len=animation_len,
                                          do_sketches=do_sketches,
                                          include_final_keyframe=do_second_keyframe
                                          )
        self.save_loc = save_loc
        self.animation_len = animation_len
        self.time = datetime.now().strftime('%m-%d-%H-%M')

        (self.save_loc / f"{self.time}").mkdir()

    def on_epoch_end(self, epoch, logs=None):
        batch = self.gen.__getitem__(index=3)
        reconstruction = self.model.predict(batch[0])

        subplot_animation(batch[0][0], batch[1][0], reconstruction[0],
                          str(self.save_loc / f"{self.time}" / f"{epoch}.png"), self.animation_len)


def load_vqvae_animation_datasets(dataloc: Path, batch_size=16, num_frames=6, do_sketches=True,
                                  do_second_keyframe=True):
    training_dataset = VQVAEAnimatedGenerator(dataloc=dataloc / "train",
                                              batch_size=batch_size,
                                              shuffle=True,
                                              animation_len=num_frames,
                                              augment=True,
                                              do_sketches=do_sketches,
                                              include_final_keyframe=do_second_keyframe)

    validation_dataset, testing_dataset = None, None
    if (dataloc / "validate").exists():
        validation_dataset = VQVAEAnimatedGenerator(dataloc=dataloc / "validate", batch_size=batch_size, shuffle=True,
                                                    animation_len=num_frames)
    if (dataloc / "test").exists():
        testing_dataset = VQVAEAnimatedGenerator(dataloc=dataloc / "test", batch_size=batch_size, shuffle=False,
                                                 animation_len=num_frames)

    return training_dataset, validation_dataset, testing_dataset


# FROM KERAS EXAMPLE OF VQVAE IMPLEMENTATION
def train_VQVAE(dataloc: Path,
                outputloc: Path = None,
                callbackloc: Path = None,
                visualize=True,
                animation=True,
                batch_size=16,
                num_frames=6,
                do_sketches=True,
                do_second_keyframe=True
                ):
    training_dataset, validation_dataset, testing_dataset = load_vqvae_animation_datasets(dataloc, batch_size,
                                                                                          num_frames=num_frames,
                                                                                          do_sketches=do_sketches,
                                                                                          do_second_keyframe=do_second_keyframe)

    training_dataset._shuffle_indices()

    vqvae_trainer = get_animation_vqvae(latent_dim=8,
                                        num_embeddings=256,
                                        num_frames=num_frames)

    ssim_loss = get_ssim_loss(num_frames)

    vqvae_trainer.compile(optimizer=tfa.optimizers.Lookahead(keras.optimizers.Adam()),
                          loss=ssim_loss)

    donkey_plotter = plotting_callback(callbackloc / "donkey" / "frames",
                                       callbackloc / "donkey" / "plots",
                                       num_frames,
                                       do_sketches=do_sketches,
                                       do_second_keyframe=do_second_keyframe)

    horse_plotter = plotting_callback(callbackloc / "horse" / "frames",
                                      callbackloc / "horse" / "plots",
                                      num_frames,
                                      do_sketches=do_sketches,
                                      do_second_keyframe=do_second_keyframe)

    vqvae_trainer.fit(training_dataset,
                      validation_data=validation_dataset,
                      epochs=100,
                      callbacks=[donkey_plotter, horse_plotter])

    encoder = vqvae_trainer.get_layer("encoder")
    quantizer = vqvae_trainer.get_layer("vector_quantizer")

    if visualize:
        test_images = testing_dataset._data_generation(0, 10)
        if animation:
            pass
            if outputloc:
                pass
        else:
            for test_image in test_images.as_numpy_iterator():
                reconstructed_image = vqvae_trainer.predict(test_image)
                show_subplot(test_image, reconstructed_image)

            # Visualizing the discrete codes
            codebook_test_images = np.squeeze(np.stack(list(test_images)))

            encoded_outputs = encoder.predict(codebook_test_images)
            flat_enc_outputs = encoded_outputs.reshape(-1, encoded_outputs.shape[-1])
            codebook_indices = quantizer.get_code_indices(flat_enc_outputs)
            codebook_indices = codebook_indices.numpy().reshape(encoded_outputs.shape[:-1])

            for i, image in enumerate(codebook_test_images):
                plt.subplot(1, 2, 1)
                plt.imshow(image.squeeze())
                plt.title("Original")
                plt.axis("off")

                plt.subplot(1, 2, 2)
                plt.imshow(codebook_indices[i])
                plt.title("Code")
                plt.axis("off")
                plt.show()

    return vqvae_trainer


def __parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_path")
    parser.add_argument("--callback_path")
    parser.add_argument("--model_path")

    args = parser.parse_args()

    return args.train_path, args.callback_path, args.model_path


if __name__ == "__main__":

    animation_length = 5
    batch_size = 16
    test_batch_size = 3

    # Variables only to set to false for ablation study.
    do_sketches = True
    do_second_keyframe = True

    train_path, callback_path, model_path = __parse_args()

    modelloc: Path = Path(model_path)
    saveloc = modelloc / datetime.now().strftime('%m-%d-%H')

    vqvae = train_VQVAE(num_frames=animation_length,
                        dataloc=Path(train_path),
                        callbackloc=Path(callback_path),
                        do_sketches=do_sketches,
                        do_second_keyframe=do_second_keyframe
                        )
    vqvae.save(str(saveloc), include_optimizer=True)
