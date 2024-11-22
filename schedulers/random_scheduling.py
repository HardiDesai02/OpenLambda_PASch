#using random scheduling


import hashlib
import random
from PIL import Image
import cv2
import pandas as pd
import numpy as np
import os
import time
import tracemalloc
import openpyxl
from openpyxl.drawing.image import Image as ExcelImage
from google.colab import files
from io import BytesIO

# Step 1: Upload images
print("Please upload your images for processing.")
uploaded = files.upload()

# Save uploaded images in "images" directory
os.makedirs("images", exist_ok=True)
for filename, data in uploaded.items():
    with open(f"images/{filename}", 'wb') as f:
        f.write(data)

# Define the grayscale conversion function
def convert_to_greyscale(image_path):
    tracemalloc.start()  # Start memory tracking
    start_time = time.perf_counter()  # Start time tracking

    # Load the image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image {image_path}. Skipping.")
        return None, None, None, None

    # Convert to grayscale
    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    execution_time = (time.perf_counter() - start_time) * 1000  # Execution time in milliseconds
    output_image_name = f"greyscale_{os.path.basename(image_path)}"
    cv2.imwrite(output_image_name, grey_image)

    # Get memory usage
    current_memory, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()  # Stop memory tracking

    # Convert memory usage to KB
    current_memory_kb = current_memory / 1024
    peak_memory_kb = peak_memory / 1024

    return output_image_name, execution_time, current_memory_kb, peak_memory_kb

# Worker class to handle tasks and cache management
class Worker:
    def __init__(self, worker_id, capacity, all_libraries):
        self.worker_id = worker_id + 1  # Use IDs from 1 to 5
        self.capacity = capacity
        self.load = 0  # Current load
        self.cache = set(random.sample(all_libraries, 2))  # Preload 2-3 random libraries
        self.last_cache_reset_time = 0  # To track when the cache was last reset
        self.core_libraries = set(all_libraries)  # Core libraries
        self.library_loading_time = 0.5  # Simulated library load time per library (in seconds)

    def is_overloaded(self, threshold):
        return self.load >= threshold

    def add_task(self, image_name, current_time, required_libraries):
        """Assign task to this worker."""
        self.load += 1
        self.cache_miss(image_name, current_time, required_libraries)

    def release_task(self):
        """Release load after task completion."""
        self.load = max(0, self.load - 1)

    def cache_miss(self, image_name, current_time, required_libraries):
        """Simulate cache miss and loading missing libraries."""
        # Simulate that each worker might need to load some libraries if not in cache
        libraries_to_load = required_libraries - self.cache
        load_time = len(libraries_to_load) * self.library_loading_time  # Time to load missing libraries

        if libraries_to_load:
            # Increment the waiting time by the library loading time
            print(f"Worker {self.worker_id}: Missing libraries {libraries_to_load}, loading time = {load_time:.2f}s")

            # Simulate the library loading delay
            time.sleep(load_time)

            # Update the cache
            self.cache.update(libraries_to_load)

        # Account for cache reset every 10 seconds
        if (current_time - self.last_cache_reset_time) >= 10:
            print(f"Worker {self.worker_id}: Cache reset after 10 seconds")
            self.cache = set(random.sample(self.core_libraries, 2))  # Reload 2 random libraries
            self.last_cache_reset_time = current_time

# Weighted random scheduler
class WeightedRandomScheduler:
    def __init__(self, workers, weights):
        self.workers = workers
        self.weights = weights  # User-defined weights for each worker

    def schedule(self, image_name, current_time, required_libraries):
        # Randomly choose a worker based on weights
        worker = random.choices(self.workers, weights=self.weights, k=1)[0]

        # Assign the task to the selected worker
        worker.add_task(image_name, current_time, required_libraries)
        return worker

# Simulate scheduling with random weighted worker assignment
def simulate_random_weighted_scheduling(image_set, scheduler, num_processes=100, total_time=50):
    process_list = []

    # Initialize worker availability to zero for each worker
    num_workers = len(scheduler.workers)
    worker_availability = [0] * num_workers  # Track when each worker will be free

    current_time = 0  # Initialize current time in seconds

    # Define required libraries for each image processing task
    all_libraries = ["hashlib", "cv2", "numpy", "os", "time", "pandas", "openpyxl"]

    while current_time < total_time and len(process_list) < num_processes:
        # Simulate image arrivals
        current_arrivals = get_current_arrivals(image_set, current_time)

        for image in current_arrivals:
            if len(process_list) >= num_processes:
                break  # Stop if we've reached the required number of processes

            # Randomly select required libraries for this image processing task
            required_libraries = set(random.sample(all_libraries, 3))  # Select 2-3 libraries

            # Schedule the image using the weighted random scheduler
            worker = scheduler.schedule(image, current_time, required_libraries)

            # Simulate greyscale conversion and track execution time
            grey_image, exec_time, current_mem, peak_mem = convert_to_greyscale(image)

            # Calculate the worker's waiting time in milliseconds
            waiting_time = max(0, (worker_availability[worker.worker_id - 1] - current_time) * 1000)  # Convert to ms

            # Update the worker's availability based on the current execution time
            worker_availability[worker.worker_id - 1] = current_time + (exec_time / 1000) + (waiting_time / 1000)  # Convert ms to seconds for time tracking

            # Add to process list
            process_list.append({
                "process_id": len(process_list) + 1,
                "color_image": image,
                "greyscale_image": grey_image,
                "arrival_time": current_time,
                "execution_time": exec_time,
                "waiting_time": waiting_time,
                "worker_id": worker.worker_id,
                "current_memory_kb": current_mem,
                "peak_memory_kb": peak_mem
            })

        # Increment current time
        current_time += 1  # Increment time in seconds

    return process_list

# Initialize workers with user-defined weights
num_workers = 5
worker_weights = [1, 2, 1, 3, 2]  # Example weights for workers
workers = [Worker(i, capacity=10, all_libraries=["hashlib", "cv2", "numpy", "os", "time", "pandas", "openpyxl"]) for i in range(num_workers)]
scheduler = WeightedRandomScheduler(workers, worker_weights)

# Prepare image list from uploaded images
image_set = [f"images/{filename}" for filename in uploaded.keys()]

# Simulate scheduling with random weighted assignment
process_list = simulate_random_weighted_scheduling(image_set, scheduler)

# Create DataFrame for results
df = pd.DataFrame(process_list, columns=[
    "process_id", "color_image", "greyscale_image", "arrival_time",
    "execution_time", "waiting_time", "worker_id",
    "current_memory_kb", "peak_memory_kb"
])

# Save results to Excel with images
df.to_excel("random_weighted_image_scheduling.xlsx", index=False)
wb = openpyxl.load_workbook("random_weighted_image_scheduling.xlsx")
ws = wb.active

# Insert thumbnails of images into Excel
for idx, row in df.iterrows():
    try:
        # Load and resize images
        color_img = Image.open(row['color_image'])
        grey_img = Image.open(row['greyscale_image'])

        # Resize images for larger thumbnails (100x100)
        color_img.thumbnail((100, 100))  # Increased size
        grey_img.thumbnail((100, 100))   # Increased size

        # Convert images to BytesIO for embedding
        color_bytes = BytesIO()
        grey_bytes = BytesIO()
        color_img.save(color_bytes, format="PNG")
        grey_img.save(grey_bytes, format="PNG")

        # Create Excel image objects
        color_image = ExcelImage(color_bytes)
        grey_image = ExcelImage(grey_bytes)

        # Set the cell values for color and grayscale images
        color_image.width = 100
        color_image.height = 50
        grey_image.width = 100
        grey_image.height = 50

        # Insert images into the corresponding cells
        ws.add_image(color_image, f'B{idx+2}')
        ws.add_image(grey_image, f'C{idx+2}')

    except Exception as e:
        print(f"Error inserting image at row {idx}: {e}")

# Save the modified workbook
wb.save("random_weighted_image_scheduling_with_images.xlsx")

# Download the result file
files.download("random_weighted_image_scheduling_with_images.xlsx")

