--A Sneaker is more similar to a Sandal than it is to a Pullover

--Define a class as an integer from 0 to 9
numberOfClasses = 10
type Class = Index numberOfClasses

--The input for the network is an image of 28 by 28 pixels
type Image = Tensor Rat [28, 28]

--The score that the classifier assigns to a class
type Score = Rat

--Declare the network
@network
classifier : Image -> Vector Score numberOfClasses

--Declare the dataset to be used
@dataset
images : Vector Image 1

--A or not A
@property
pulloverLowScore : Bool
pulloverLowScore = 
    let scores = classifier (images ! 0) in
    scores ! 7 > scores ! 2 or not(scores ! 7 > scores ! 2)