--Statement
--A Sneaker is more similar to a Sandal than it is to a Pullover

numberOfClasses = 10

type Class = Index numberOfClasses

pullover = 2
sandal = 5
sneaker = 7

type Image = Tensor Rat [28, 28]

type Score = Rat

@network
classifier : Image -> Vector Score numberOfClasses

@parameter(infer=True)
n : Nat

@dataset
trainingImages : Vector Image n

--Define a valid image, i.e. one that has pixel values between 0 and 1
validPixel : Rat -> Bool
validPixel p = 0 <= p <= 1

validImage : Image -> Bool
validImage img = forall i j . validPixel (img ! i ! j)

--what is the score of a class in an image
score : Image -> Class -> Score
score img class =
    let scores = classifier img in
    scores ! class

--is the top score for the image sneaker
firstChoiceSneaker : Image -> Bool
firstChoiceSneaker img =
    let scores = classifier img in
    forall class . class != sneaker => scores ! sneaker > scores ! class

--need idx with sneakers in it
--forall vs foreach?
@property
pulloverLowScore : Vector Bool n
pulloverLowScore =
    foreach i . (firstChoiceSneaker (trainingImages ! i) and validImage (trainingImages ! i)) 
        => score (trainingImages ! i) sandal > score (trainingImages ! i) pullover