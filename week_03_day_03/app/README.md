# Setup
1. Download the **encoder.pth** file [here](https://drive.google.com/file/d/1-_5rvRgYxAisnKpf9dO8HD2ObQZaka_A/view?usp=share_link).
2. Move the **encoder.pth** file to the **encoders/default** folder.
3. Build the Docker image: `$ docker built -t aliimtegar/fruit-recognition:1.0 .`
4. Run Docker Compose: `$ docker-compose up`
5. Go to [localhost:8000/docs](http://localhost:8000/docs) to start trying the API!

# API
## /register
Registers the vector embedding of an image to the database.
- Method: POST
- Request Parameters:
  - **label**: The label of the image.
  - **file**: The image file of fruit(s).
- Response JSON:
  - **message**: A message indicating whether the vector embedding of the image was registered successfully.

## /recognize
Recognizes the label of an image.
- Method: POST
- Request Parameters:
  - **file**: The image file of fruit(s).
- Response JSON:
  - **label**: The recognized label of the image.

# Screenshots
![Screenshot 1](https://i.ibb.co/tMfMpKZ/Screenshot-2023-03-09-at-15-51-44.png)
![Screenshot 2](https://i.ibb.co/6NKzr2T/Screenshot-2023-03-09-at-15-52-22.png)
![Screenshot 3](https://i.ibb.co/GsSC8W7/Screenshot-2023-03-09-at-15-53-53.png)
![Screenshot 4](https://i.ibb.co/92GpZVh/Screenshot-2023-03-09-at-15-54-30.png)

# Dataset
The [Even More Fruitssssss - [Image Dataset]
](https://www.kaggle.com/datasets/yash161101/even-more-fruitssssss) dataset was used in this project.