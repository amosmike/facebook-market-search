# facebook-market-search
## Introduction
This is a capstone project for AiCore that aims to create a replica of the Facebook Marketplace Search Ranking using machine learning and machine learning engineering skills. The project uses various technologies such as Python, FastAPI, Math, NumPy, Pandas, Pillow, Pydantic, Python-Multipart, PyTorch, Pytz, Transformers, and Uvicorn.

## Milestone 1 - Overview of the System
The project is divided into five milestones. The first milestone involved the exploring the provided dataset by connecting to the provided sandbox environment, setting up the environment in the EC2 instance using the content in the S3 buckets provided, and creating a pipeline using Pandas, NumPy, and Math to remove rows with missing data and clean the tabular data. A second pipeline is created to clean the image data provided.

## Milestone 2 - Explore the Dataset
The second milestone involves creating two simple machine learning models, a regression model to predict the price of products based on tabular data and a classification model to categorize product images associated with products. The models required manipulation of the cleaned data sets to fit with the models, including encoding of categorical variables. The models perform poorly but demonstrate the pipelines/implemented methods to create them.

## Milestone 3 - Create the Vision Model then Feature Extraction Model
The third milestone is focused on creating a vision model. The goal is to create and train a neural network to ingest images and classify the category the product belongs to. First, the images retrieved are converted to a PyTorch dataset, in the form of tensors, to be ingested into a neural network. A simple convolutional neural network is created, along with a training loop that processes batches in the given number of epochs. To try and improve the model, transfer learning is applied using a ResNet50 model from the models library. Finally, to enable the future used of a trained model, and monitoring of model performance, details and weights of the models are saved for each epoch and also the final model.

## Milestone 4 - Creat Search Index using FAISS
The fourth milestone was to create a search index using FAISS. For this a feature extractor model is created to return a json with the image id as keys and the image tensor from 'layers.fc.1' as the values. This json is then used in the FAISS model with the faiss.IndexFlatL2 to find images with a like features.  

## Milestone 5 - Configure and Deploy the Model Serving API
The fifth and final milestone involves configuring and deploying the model serving API. A script is written to create an API using FastAPI to take an image and return the predicted category with numerical confidence. This is done using the trained weights from the neural network model created in milestone 3. A Dcokerfile has also been produced for easy deployement.