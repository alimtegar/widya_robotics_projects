import os
import argparse
import shutil
from PIL import Image

def convert_images(input_dir, output_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Loop through all files in the input directory
    for filename in os.listdir(input_dir):
        # Check if the file is an image
        if filename.endswith(".png") or filename.endswith(".bmp") or filename.endswith(".gif") or filename.endswith(".webp"):
            # Open the image using PIL/Pillow
            image = Image.open(os.path.join(input_dir, filename))
            # Convert the image to RGBA if it has an alpha channel
            if image.mode == "RGBA":
                background = Image.new("RGBA", image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[3])
                converted = background.convert("RGB")
            elif image.mode == "P":
                # Convert the image to RGBA
                converted = image.convert("RGBA")
                background = Image.new("RGBA", converted.size, (255, 255, 255))
                background.paste(converted, mask=converted.split()[3])
                converted = background.convert("RGB")
            else:
                # Convert the image to RGB
                converted = image.convert("RGB")
            # Save the image as a JPG in the output directory
            output_filename = os.path.splitext(filename)[0] + ".jpg"
            output_path = os.path.join(output_dir, output_filename)
            try:
                converted.save(output_path)
            except:
                # If the image cannot be converted to JPG, print an error message and skip the file
                print(f"Error converting {filename} to JPG")
        else:
            # Copy the file to the output directory
            output_path = os.path.join(output_dir, filename)
            shutil.copy2(os.path.join(input_dir, filename), output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert images to JPG format")
    parser.add_argument("input_dir", help="path to input directory")
    parser.add_argument("output_dir", help="path to output directory")
    args = parser.parse_args()

    convert_images(args.input_dir, args.output_dir)
