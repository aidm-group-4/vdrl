from glob import glob
import os
import re

sku_regex = r"^\d{5}"

mood_dir = (
    "/Users/aditwhorra/Downloads/Presentation Master v8 - Complete/Images/Moodshots"
)
white_dir = "/Users/aditwhorra/Downloads/Presentation Master v8 - Complete/Images/White Background"


mood_files = glob(os.path.join(mood_dir, "*.jpg"))
mood_file_names = list(map(lambda x: x.split("/")[-1], mood_files))
mood_images = list(set(list(map(lambda x: re.match(sku_regex, x)[0], mood_file_names))))

white_files = glob(os.path.join(white_dir, "*.jpg"))
white_file_names = list(map(lambda x: x.split("/")[-1], white_files))
white_images = list(set(list(map(lambda x: re.match(sku_regex, x)[0], white_file_names))))

with open("/Users/aditwhorra/Desktop/mood_images.txt", "w") as file:
    for string in mood_images:
        file.write(string + "\n")

with open("/Users/aditwhorra/Desktop/white_images.txt", "w") as file:
    for string in white_images:
        file.write(string + "\n")
