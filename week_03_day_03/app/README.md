# Setup
1. Download the **checkpoint** file [here](https://drive.google.com/file/d/1-4OhgH7N-AQBFHa05HRq3MD7CG42ojAm/view?usp=sharing).
2. Move the **checkpoint** file to the **models** folder.
3. Build the Docker image: `$ docker built -t aliimtegar/nuclei-segmentation:1.0 .`
4. Run Docker Compose: `$ docker-compose up`
5. Go to [localhost:8000/docs](http://localhost:8000/docs) to start trying the API!

# API
## /segment-nuclei
- Method: POST
- Request Parameters:
  - **file**: The image file of nuclei.
- Response JSON:
  - **mask**: The base64-encoded image of the mask.

# Dataset
The [2018 Data Science Bowl](https://www.kaggle.com/competitions/data-science-bowl-2018) dataset by Booz Allen Hamilton was used in this project.