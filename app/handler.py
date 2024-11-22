import hashlib
import random
import time
import tracemalloc
import cv2
import pandas as pd
import numpy as np
import os
import openpyxl
from openpyxl.drawing.image import Image as ExcelImage
from PIL import Image
from io import BytesIO

# Define the grayscale conversion function
def convert_to_greyscale(image_path):
    tracemalloc.start()  # Start memory tracking
    start_time = time.perf_counter()  # Start time tracking

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image {image_path}. Skipping.")
        return None, None, None, None

    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    execution_time = (time.perf_counter() - start_time) * 1000  # Execution time in milliseconds
    output_image_name = f"greyscale_{os.path.basename(image_path)}"
    cv2.imwrite(output_image_name, grey_image)

    current_memory, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()  # Stop memory tracking

    current_memory_kb = current_memory / 1024
    peak_memory_kb = peak_memory / 1024

    return output_image_name, execution_time, current_memory_kb, peak_memory_kb

# Worker class for handling tasks
class Worker:
    def __init__(self, worker_id, capacity, all_libraries):
        self.worker_id = worker_id + 1
        self.capacity = capacity
        self.load = 0
        self.cache = set(random.sample(all_libraries, 2))
        self.last_cache_reset_time = 0
        self.core_libraries = set(all_libraries)
        self.library_loading_time = 0.5

    def is_overloaded(self, threshold):
        return self.load >= threshold

    def add_task(self, image_name, current_time, required_libraries):
        self.load += 1
        self.cache_miss(image_name, current_time, required_libraries)

    def release_task(self):
        self.load = max(0, self.load - 1)

    def cache_miss(self, image_name, current_time, required_libraries):
        libraries_to_load = required_libraries - self.cache
        load_time = len(libraries_to_load) * self.library_loading_time

        if libraries_to_load:
            print(f"Worker {self.worker_id}: Missing libraries {libraries_to_load}, loading time = {load_time:.2f}s")
            time.sleep(load_time)
            self.cache.update(libraries_to_load)

        if (current_time - self.last_cache_reset_time) >= 10:
            print(f"Worker {self.worker_id}: Cache reset after 10 seconds")
            self.cache = set(random.sample(self.core_libraries, 2))
            self.last_cache_reset_time = current_time

# Scheduler for image processing tasks
class PASchScheduler:
    def __init__(self, workers, overload_threshold):
        self.workers = workers
        self.overload_threshold = overload_threshold

    def schedule(self, image_name, current_time, required_libraries):
        affinity_worker_ids = [
            consistent_hash(image_name, len(self.workers)),
            consistent_hash(image_name + "salt", len(self.workers))
        ]
        affinity_workers = [self.workers[i] for i in affinity_worker_ids]
        target_worker = min(affinity_workers, key=lambda w: w.load)

        if target_worker.is_overloaded(self.overload_threshold):
            target_worker = min(self.workers, key=lambda w: w.load)

        target_worker.add_task(image_name, current_time, required_libraries)
        return target_worker

def consistent_hash(item, num_buckets):
    return int(hashlib.md5(item.encode()).hexdigest(), 16) % num_buckets

# Simulate image scheduling
def get_current_arrivals(image_set, current_time, max_arrivals=4):
    num_new_images = random.randint(1, max_arrivals)
    return random.sample(image_set, min(num_new_images, len(image_set)))

def simulate_scheduling(image_set, scheduler, num_processes=100, total_time=50):
    process_list = []
    worker_availability = [0] * len(scheduler.workers)
    current_time = 0
    all_libraries = ["hashlib", "cv2", "numpy", "os", "time", "pandas", "openpyxl"]

    while current_time < total_time and len(process_list) < num_processes:
        current_arrivals = get_current_arrivals(image_set, current_time)

        for image in current_arrivals:
            if len(process_list) >= num_processes:
                break

            required_libraries = set(random.sample(all_libraries, 3))
            worker = scheduler.schedule(image, current_time, required_libraries)

            grey_image, exec_time, current_mem, peak_mem = convert_to_greyscale(image)
            waiting_time = max(0, (worker_availability[worker.worker_id - 1] - current_time) * 1000)

            worker_availability[worker.worker_id - 1] = current_time + (exec_time / 1000) + (waiting_time / 1000)
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

        current_time += 1

    return process_list

# Initialize workers and scheduler
num_workers = 5
workers = [Worker(i, capacity=10, all_libraries=["hashlib", "cv2", "numpy", "os", "time", "pandas", "openpyxl"]) for i in range(num_workers)]
scheduler = PASchScheduler(workers, overload_threshold=8)

# Gather images
image_set = [f"/app/images/{filename}" for filename in os.listdir("/app/images") if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg")]


# Run scheduling and save results
process_list = simulate_scheduling(image_set, scheduler)
df = pd.DataFrame(process_list)

if not df.empty:
    df.to_excel("/app/image_scheduled_worker_nodes_with_images.xlsx", index=False)
    wb = openpyxl.load_workbook("/app/image_scheduled_worker_nodes_with_images.xlsx")
    ws = wb.active

    for idx, row in df.iterrows():
        try:
            color_img = Image.open(row['color_image'])
            grey_img = Image.open(row['greyscale_image'])

            color_img.thumbnail((100, 100))
            grey_img.thumbnail((100, 100))

            color_bytes = BytesIO()
            grey_bytes = BytesIO()
            color_img.save(color_bytes, format="PNG")
            grey_img.save(grey_bytes, format="PNG")

            color_image = ExcelImage(color_bytes)
            grey_image = ExcelImage(grey_bytes)

            color_image.width = 100
            color_image.height = 50
            grey_image.width = 100
            grey_image.height = 50

            ws.add_image(color_image, f'B{idx+2}')
            ws.add_image(grey_image, f'C{idx+2}')

        except Exception as e:
            print(f"Error inserting image at row {idx}: {e}")

    wb.save("/app/image_scheduled_worker_nodes_with_images.xlsx")
else:
    print("No data available to save.")

