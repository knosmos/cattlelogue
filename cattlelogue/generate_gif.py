import imageio
import os
import click
from tqdm import tqdm


@click.command()
@click.option("--prefix", type=str, default="crop", help="Prefix for the image files")
def generate_gif(prefix):
    """
    Generate a GIF from a series of images with the specified prefix.
    """
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    filenames = sorted(
        [
            os.path.join(OUTPUT_DIR, f)
            for f in os.listdir(OUTPUT_DIR)
            if f.startswith(prefix) and f.endswith(".png")
        ]
    )

    if not filenames:
        print("No images found with the specified prefix.")
        return

    # Create a GIF from the images
    with imageio.get_writer(
        os.path.join(OUTPUT_DIR, f"{prefix}.gif"), mode="I", loop=0
    ) as writer:
        for filename in tqdm(filenames):
            image = imageio.imread(filename)
            writer.append_data(image)

    print(f"GIF saved as {os.path.join(OUTPUT_DIR, f'{prefix}.gif')}")


if __name__ == "__main__":
    generate_gif()
