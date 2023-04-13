# Deep Learning Meeting Minutes 

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
