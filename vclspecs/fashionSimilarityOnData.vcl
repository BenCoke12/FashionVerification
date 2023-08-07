--A Sneaker is more similar to a Sandal than it is to a Pullover

--Define a class as an integer from 0 to 9
numberOfClasses = 10
type Class = Index numberOfClasses

--The indecies of the classes that are considered for this property
pullover = 2
sandal = 5
sneaker = 7
ankleBoot = 9

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
images : Vector Image n

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

--Is the score for sandal higher than the score for ankleboot
sandalGreaterThanAnkleBoot : Image -> Bool
sandalGreaterThanAnkleBoot image =
    score image sandal > score image ankleBoot

--Is the score for sandal higher than the score for pullover
sandalGreaterThanPullover : Image -> Bool
sandalGreaterThanPullover image =
    score image sandal > score image pullover

--Check that if the first choice of class for the image is Sneaker then the score
--for Sandal is greater than the score for Pullover
similaritySneakerSandal : Image -> Bool
similaritySneakerSandal image = 
    firstChoiceSneaker image => sandalGreaterThanPullover image

--Check the property for every image in the given dataset
@property
pulloverLowScore : Vector Bool n
pulloverLowScore = 
    foreach i . similaritySneakerSandal (images ! i)