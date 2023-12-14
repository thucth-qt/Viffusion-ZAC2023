from PIL import ImageOps

def pad(image, desired_width=1024, desired_height=533, color="white"):

    padding_width = (desired_width - image.width) // 2
    padding_height = (desired_height - image.height) // 2

    padded_image = ImageOps.expand(image, border=(padding_width, padding_height), fill=color)
    if padded_image.size!=(desired_width, desired_height):
        padded_image = padded_image.resize((desired_width, desired_height))
    return padded_image
