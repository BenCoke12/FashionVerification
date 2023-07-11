--A Sneaker is more similar to a Sandal than it is to a Pullover

--Define a class as an integer from 0 to 9
numberOfClasses = 10
type Class = Index numberOfClasses

--The indecies of the classes that are considered for this property
pullover = 2
sandal = 5
sneaker = 7

--The input for the network is an image of 28 by 28 pixels
type Image = Tensor Rat [28, 28]

--The score that the classifier assigns to a class
type Score = Rat

--Declare the network
@network
classifier : Image -> Vector Score numberOfClasses


@parameter(infer=True)
n : Nat

--Declare the dataset to be used
@dataset
trainingImages : Vector Image n

--Define a valid image, i.e. one that has pixel values between 0 and 1
validPixel : Rat -> Bool
validPixel p = 0 <= p <= 1

validImage : Image -> Bool
validImage img = forall i j . validPixel (img ! i ! j)

--What is the score of a class in an image
score : Image -> Class -> Score
score img class =
    let scores = classifier img in
    scores ! class

--Is the top score for the image the class Sneaker
firstChoiceSneaker : Image -> Bool
firstChoiceSneaker img =
    let scores = classifier img in
    forall class . class != sneaker => scores ! sneaker > scores ! class


--Check that for every i in the dataset, if the first choice of class is Sneaker
--and the image is a valid image then -> The score for Sandal is higher than the 
--score for Pullover.
@property
pulloverLowScore : Vector Bool n
pulloverLowScore =
    foreach i . (firstChoiceSneaker (trainingImages ! i) and validImage (trainingImages ! i)) 
        => score (trainingImages ! i) sandal > score (trainingImages ! i) pullover