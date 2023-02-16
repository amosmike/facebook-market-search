#%%
import os
from tqdm import tqdm
from PIL import Image

def resize_image(final_size, im):
    """
    Function to normalised the size and mode of the image to the input final size and RGB mode.
    Args:
        final_size (int): integer value of the final pixel size of the new image.
        image (image): image object of desired image to normalize.
    Returns:
        image: image object of original image to formated to desired pixel size in RGB mode.

    """
    size = im.size
    ratio = float(final_size) / max(size)
    new_image_size = tuple([int(x*ratio) for x in size])
    im = im.resize(new_image_size, Image.ANTIALIAS)
    new_im = Image.new("RGB", (final_size, final_size))
    new_im.paste(im, ((final_size-new_image_size[0])//2, (final_size-new_image_size[1])//2))
    return new_im

if __name__ == '__main__':
    path = "images/"
    dirs = os.listdir(path)
    print(len(dirs))
    final_size = 512
    # for n, item in enumerate(dirs[:5], 1):
    if os.path.exists("cleaned_images/")==False:
            os.makedirs("cleaned_images/")

    for n, item in tqdm(enumerate(dirs, 1), total=len(dirs)):
        if item[-4:]=='.jpg':
            filename=str(item)
            # filename = f"{str(item[:-4])}_resized_to_{final_size}.jpg"
            im = Image.open('images/' + item)
            new_im = resize_image(final_size, im)
            new_im.save('cleaned_images/' + filename)