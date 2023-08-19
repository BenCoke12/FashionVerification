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

--is the score for sneaker over a threshold
highScoreSneaker : Image -> Bool
highScoreSneaker image =
    let scores = classifier image in
    scores ! sneaker > 10

--is the image within epsilon distance of a sneaker image
withinEpsilon : Image -> Image -> Bool
withinEpsilon img sneakerImage = 
    forall i j . ((sneakerImage ! i ! j) - 0.06) <= img ! i ! j <= ((sneakerImage ! i ! j) + 0.06)

@dataset
imageData : Vector Image 1

@property
pulloverLowScore : Bool
pulloverLowScore = forall img . 
    (validImage img and firstChoiceSneaker img and withinEpsilon img (imageData ! 0))
        => score img sandal + 1 > score img pullover
