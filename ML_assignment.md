# ArtLabs ML Engineer Assignment

The goal of the task is to create a very basic image classification engine where a user sends an image and receives a response which is a prediction of the ML model.


## Objective

Based on the requirements below, you will define the adequate data model, a web server and its API.
The past prediction history should be stored in a local database, for the sake of this exercise, we can use an sqlite database.

## Application server

The application server has a client-facing API, we need the following endpoints:

### POST /api/train

Trains the model on the data stored on the machine, you can define how you store training data on the machine yourself. You can choose any image classification model you want, it is better you choose a CNN architecture.

Request:

```
none
```

Response:

```
{
  "success": true
}
```

### GET /api/predict

Downloads the image from the provided link and predicts with a trained model. After the prediction is done the prediction metadata is stored in the database for the "/api/get_past_predictions/" request.

Request:

```
{
    "img_link": <img_link>
}
```

Response:

```
{
    "prediction": <label>
}
```

### GET /api/get_past_predictions/

Returns all the past model predictions.

Request:

```
none
```

Response:

```
{
  "predictions": [
    {
        "prediction_time": <time>,
        "image_link": <link>,
        "image_label": <label>
    },
    ...
  ]
}
```

### DELETE /api/clear_past_predictions

Clears the database with all the past model predictions.

Request:

```
none
```

Response: status code 204

# Miscellaneous

This take-home is designed to take 2 to 8 hours. It would be great if you write tests for your engine as well!
All the best, ArtLabs team.
