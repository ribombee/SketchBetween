import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation

# Plotting
def show_subplot(original, reconstructed):
    plt.subplot(1, 2, 1)
    plt.imshow(original.squeeze())
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed.squeeze())
    plt.title("Reconstructed")
    plt.axis("off")

    plt.show()

def plot_animation(frames, save=False):

    fig, ax = plt.subplots()

    ims = []

    for frame in frames:
        im = ax.imshow(frame, animated=True)
        ims.append([im])

    animation = ArtistAnimation(fig, ims, interval=200, blit=True, repeat_delay=500)

    #TODO: you can save the animation

    plt.show()

def subplot_animation(input, y_true, y_pred, save_loc: str, animation_len: int):

    if y_true is not None:
        fig, ax = plt.subplots(3, animation_len)

        for frame_idx in range(animation_len):
            ax[0, frame_idx].imshow(input[frame_idx])
            ax[1, frame_idx].imshow(y_true[frame_idx])
            ax[2, frame_idx].imshow(y_pred[frame_idx])
        plt.savefig(
            save_loc,
            dpi=300)
        plt.close(fig)

    else:
        fig, ax = plt.subplots(2, animation_len)

        for frame_idx in range(animation_len):
            ax[0, frame_idx].imshow(input[frame_idx])
            ax[1, frame_idx].imshow(y_pred[frame_idx])
        plt.savefig(
            save_loc,
            dpi=300)
        plt.close(fig)