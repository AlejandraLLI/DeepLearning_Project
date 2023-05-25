# Deep Learning Meeting Minutes 

## May 25th meeting

### Discussion:
- Scratch models are finished. We just need to check Sam's branch...
- Pre-trained model experiments are finished. 
- Transfer learning experiments are finished. Don't forget to justify selection of EfficientNet in ppt. 
- Need to test the best of each model on a test set. 
- Style Transfer experiments and model justification are finished. 
- We will agree on time to finish the project on Saturday, after finishing Cloud. 

### Tasks for weekend
- Run best model for test set.
- Run best models for out of sample images (our headshots & volunteers)
- Write report (ppt) and presentation. 


## May 18th meeting

### Discussion:
- Showed style transfer experiments. Aging is definitely not working. We will transfer style of cartoons. Pick the best for every team member for the ppt. 
- Showed results for almost all scratch models. Data augmentation does make a difference. 
- Showed results for a pre-trained model. 
- Still need to work on transfer learning. Boss has had issues with server, we might need to help him run the models. 
- Ask class if we can have their pictures (baby and recent). It would be better to have a folder for the test images to run all predictions on the same data. 
- Discusseed presentation vs report. We agreed on one "full" presentation an one "short" presentation for the expo.

### Tasks for next week 
- Start drafting full presention and short presentation (ALL)
- Ask class for volunteers and Ashish and make a folder of all out of sample tests (Ruben)
- Finish running age prediction from scratch models (SAM)
- Train transfer learning (BOSS)
- Compare different models for style transfer (ALE)
- Clarify requirements with Ashish (ALL)

## May 11th meeting

### Discussion:
- Reviewed Sam's models. 
- Most of the team didn't have time to work on Deep Learning. We will focus more on it this week. 

### Tasks for next week 
- Age prediction from scratch original vs augmented (Sam)
- Age prediction with pre-trained model. (Ruben)
- Age prediction with re-training a pre-trained model (Boss)
- Style transfer (Ale): 
		- Give a last try to age transfering 
		- Use different loss functions to generate transfer.

## May 4th meeting

### Discussion:
- Check data agumentation process. Need to adjust some probabilities for transformations. 
- To check how many GPUs in deep dish server use this: 

`import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))`

- Discusses possible style transfer solutions. Need to further research about it. 

### Tasks for next week 
- Change probability of rotation and horiztonal flip to a higher number (around 0.3)
- Test different architectures with the augmented data (Sam)
- Look for alternative pre-trained models that are more suitable to aging (Ruben). 
- Look for other tutorials on style transfering and compare (Ale)
- Try to find out if there is a way to do style transfering from scratch (Boss)

## April 27th meeting

### Discussion:

- Ruben has worked on the data augmentation. We need to define which transformations we will use from the package. 
- We have a first DNN. Need to try building different architectures and parameters. 
- For Style Transfer, there seems to be no "state of the art code" and that the common thing to do is to use a pre-trained model. We need to ask the professor about this. 
- Might want to put all the cloud engineering learning (container, environment, pep8 etc) in practice in this project if we have time (and energy!!)

### Tasks for next week: 

- Ask Ashish about which transformations we need to use and finish data augmentation (Ruben)
- Test different architectures (SAM)
- Figure out how to run it in deep dish and Class balance (Boss)
- Keep working on the style transfering (Ale) 

## April 20th meeting

### Discussion

- Data Augmentation: data is not uniformly spread by age. Need to incorporate flipping, contrast, corpping. 
- Problem 1: age prediction (regresssion)
- Problem 2: style transfering (making people look older/younger)

### Tasks for next week: 

- Look into data augmentation (Ruben)
- Transform images into numbers (Sam)
- Problem 1: look for documentation/tutorials (Boss)
- Problem 2: look for documentation/tutorials (ALE)

## April 13th meeting

### Discussion

Snakes:

	- Pros:
		- well defined problem, can have a practical application, 
		- train set with 5,508 images
		- 35 classes to predict. 
		- Some notebooks to get inspired 

	- Cons:
		- Not sure of the second problem ... would need professors input.

	- Applications: 
		- classification
		- compression?? image restoration?? fake detection?? 
		- Ashsish said that we could alter the image and ask the model to complete the missing part ("in-painting").


Chest x-ray: 

	- Pros: 
		- Well defined problem, practical application
		- Many notebooks to get inspired 
		- Binary problem 

	- Cons: 
		- Less images but heavier (high resolution)

	- Applications: 
		- Classification
		- complete the image???

Brain tumor:

	-Pros: 
		- Well defined problem, practical application
		- Many more notebooks to get inspired 
		- 4 categories: no tumor, pituitary, glioma, meningioma
		- 100 images per class. 

	-Cons: 
		- Less images but heavier (high resolution)

	-Applications: 
		- Classification
		- complete the image???


Celebrities: 

	- Pros:
		- 202,599 faces of celebrities

	- Cons: 
		- More difficult to define an interesting problem.
		- Multiple data sets of attributes. Cleaning might be complicated.
		- sample notebooks

	- Applications: 
		- Face recognition (although I think it does not include names)
		- Classification of physical characteristics 


Aging: 

	- Pros: 
		- 99 folders (ages) with approx 100 images each... 

	- Cons: 
		- dat set is generated, might not capture real patterns
		- sample size varies by age 
		- sample notebooks 

	- Applications: 
		- Predict age range of a person
		- Predict emotions (might need labeling and/or another dataset)
		- Style-transfer for aging or switch gender. (Ashish gave this link for emotions training https://paperswithcode.com/dataset/aff-wild)


Military Aircraft:

	- Pros:
		- can define a rea world problem
		- 43 types of aircrafts
		- At least 200 images per type
		- sample notebooks

	- Cons:
		- Not sure what the "annotated samples mean", 
		- Has extra data set with attributes, might be difficult to bind.
		- different sample size for each type 
		- not sure about the second problem

	- Applications: 
		- Classification 
		- ???? 


### Tasks for next week: 

- Top 2 data sets: snakes & ages (prefered)
- Talk to Ashish to discuss applications, pros and cons. 
- Research what is EDA on images.
- Split EDA when the data set is defined. 



## April 6th meeting

### Discussion

Data Options:

- Snake Id 2
https://www.kaggle.com/datasets/oossiiris/hackerearth-deep-learning-identify-the-snake-breed

- Celebrity/Age Detection 3
https://www.kaggle.com/datasets/jessicali9530/celeba-dataset
https://www.kaggle.com/datasets/frabbisw/facial-age

- Healthcare (Chest x-ray, brain tumor) 1
https://www.kaggle.com/datasets/tolgadincer/labeled-chest-xray-images
https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri

- Military plane id  4
https://www.kaggle.com/datasets/a2015003713/militaryaircraftdetectiondataset

Snake and Healthcare tied in preference, then Celebrity/age and then military airplanes. 

### Tasks for next week

- Check the 3 data sets for Deep leanring, find pros and cons, and try to define the two problems
