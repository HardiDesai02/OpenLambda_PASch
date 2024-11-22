# OpenLambda_PASch

This project uses the [OpenLambda](https://github.com/open-lambda/open-lambda) framework by adding functionality to process images using worker nodes, manage task scheduling using PASch(Package Aware Schedular) Algorithm, and output results to an Excel file with embedded image thumbnails.

---

## Features
1. **Image Processing:** Convert images to grayscale and capture execution metrics like memory and time.
2. **Worker Node Simulation:** Implements worker nodes with task scheduling and cache management.
3. **Output to Excel:** Saves results with embedded grayscale images into an Excel file.

---

## Prerequisites
### Tools Required
1. [Git](https://git-scm.com/downloads)
2. [Docker](https://www.docker.com/products/docker-desktop/)
3. Python 3.x with `pip` (only if running locally without Docker)

---

## Installation and Setup

### Step 1: Clone the OpenLambda Repository
First, clone the original OpenLambda repository:
```bash
git clone https://github.com/open-lambda/open-lambda.git
cd open-lambda
```
### Step 2: Replace/Add Modified Files
Replace or add the following modified files to the `open-lambda` folder:

1. **`app/handler.py`:** Contains the core application logic for image processing.
2. **`Dockerfile`:** Specifies the Docker environment for the project.
3. **`requirements.txt`:** Lists all Python dependencies.

To copy these files from your local directory (assuming they are in a folder called `modified_files`), run the following command:
```bash
cp -r ../modified_files/* .
```
Ensure the following:

- The `handler.py` file is located inside the `app/` directory (`open-lambda/app/handler.py`).
- The `Dockerfile` and `requirements.txt` are placed in the root of the `open-lambda` folder.

---

### Step 3: Install Dependencies (Optional for Local Execution)
If you wish to run the application locally instead of using Docker, you need to install the required Python dependencies. Use the following commands:

```bash
pip install -r requirements.txt
```
### Step 4: Build the Docker Image
To create a Docker image for the project, run the following command from the `open-lambda` directory:

```bash
docker build -t worker_nodes .
```
### Step 5: Run the Application
Now that the Docker image has been built, you can run the application to process the images. Use the following command to start the Docker container:

```bash
docker run --rm -v $(pwd)/images:/app/images worker_nodes
```
### Step 6: Retrieve the Output File
After the processing is complete, the Excel file (`image_scheduled_worker_nodes_with_images.xlsx`) will be available inside the container. To retrieve it, follow these steps:

1. **Run container with specific name**:

   ```bash
   docker run --name temp_worker_nodes -v $(pwd)/images:/app/images worker_nodes
    ```
2. **Copy the output Excel file from the container to your local machine:**:

    ```bash
    docker cp temp_worker_nodes:/app/image_scheduled_worker_nodes_with_images.xlsx .
    ```
3. **Remove the Container:**:
      ```bash
      docker rm temp_worker_nodes
      ```
The Excel file image_scheduled_worker_nodes_with_images.xlsx will now be available in your current directory.

## Conclusion

This project demonstrates how to extend the OpenLambda framework to process images using Docker. The images are processed in worker nodes, and the results are saved to an Excel file containing both the original and grayscale images, as well as performance metrics. By following the steps above, you can easily deploy this image processing application using Docker, retrieve the results, and troubleshoot any common issues that may arise during the process.

