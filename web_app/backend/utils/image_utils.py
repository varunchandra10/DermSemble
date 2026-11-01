import io
import numpy as np
from PIL import Image, UnidentifiedImageError
from backend.utils.exceptions import InvalidImageError, UnsupportedFormatError

def bytes_to_pil(image_bytes: bytes) -> Image.Image:
    """
    Converts raw bytes to a PIL Image.
    Raises specific errors for invalid or unsupported formats.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()  # Verify it's a valid image
        
        # Need to reopen because verify() closes the file stream
        img = Image.open(io.BytesIO(image_bytes))
        
        if img.format not in ['JPEG', 'PNG']:
            raise UnsupportedFormatError(f"Unsupported format: {img.format}. Allowed: JPEG, PNG.")
            
        return img
    except UnidentifiedImageError:
        raise InvalidImageError("Could not identify the image file.")
    except Exception as e:
        if isinstance(e, UnsupportedFormatError):
            raise
        raise InvalidImageError(f"Error processing image bytes: {str(e)}")

def pil_to_numpy(img: Image.Image) -> np.ndarray:
    """
    Converts a PIL Image to a numpy array.
    """
    return np.array(img)

def ensure_rgb(img: Image.Image) -> Image.Image:
    """
    Ensures the image is in RGB format, converting it if necessary.
    Handles RGBA (removing alpha) and Grayscale.
    """
    if img.mode != 'RGB':
        if img.mode == 'RGBA':
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            return background
        else:
            return img.convert('RGB')
    return img

def resize_with_padding(img: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    """
    Resizes an image preserving aspect ratio, padding the rest with black.
    """
    img_ratio = img.width / img.height
    target_ratio = target_size[0] / target_size[1]

    if target_ratio > img_ratio:
        # It must be fixed by height
        resize_height = target_size[1]
        resize_width = int(round(resize_height * img_ratio))
    else:
        # Fixed by width
        resize_width = target_size[0]
        resize_height = int(round(resize_width / img_ratio))

    image_resized = img.resize((resize_width, resize_height), Image.Resampling.LANCZOS)
    
    background = Image.new('RGB', target_size, (0, 0, 0))
    offset = (
        round((target_size[0] - resize_width) / 2),
        round((target_size[1] - resize_height) / 2)
    )
    background.paste(image_resized, offset)
    return background
