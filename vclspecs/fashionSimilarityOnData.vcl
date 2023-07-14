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

--Is the score for sandal higher than the score for pullover
sandalGreaterThanPullover : Image -> Bool
sandalGreaterThanPullover image =
    score image sandal > score image pullover

fullProperty : Image -> Bool
fullProperty image = 
    firstChoiceSneaker image => False
    
    --firstChoiceSneaker image => sandalGreaterThanPullover image
    --(firstChoiceSneaker image and sandalGreaterThanPullover image) or (not(firstChoiceSneaker image) and sandalGreaterThanPullover image) or (not(firstChoiceSneaker image) and not(sandalGreaterThanPullover image))
    --True => sandalGreaterThanPullover image
    --False => False
    --firstChoiceSneaker image and sandalGreaterThanPullover image
    -- to find first choice sneaker but pullover greater than sandal use:
    --firstChoiceSneaker image and not sandalGreaterThanPullover image
    -- => True
    --sandalGreaterThanPullover image

--Check that for every i in the dataset, if the first choice of class is Sneaker
--and the image is a valid image then -> The score for Sandal is higher than the
--score for Pullover.
@property
pulloverLowScore : Vector Bool n
pulloverLowScore = 
    foreach i . fullProperty (images ! i)