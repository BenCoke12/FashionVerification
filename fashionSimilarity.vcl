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

@property
pulloverLowScore : Bool
pulloverLowScore = forall img . (validImage img and firstChoiceSneaker img) => score img sandal > score img pullover
