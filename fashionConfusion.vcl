--Sneakers() and sandals() are one group, coats() and pullovers() are another
--In-group confusion is ok, check that there is no cross-group confusion

numberOfClasses = 10

--Item is a type of index over the number of classes in the dataset
type Class = Index numberOfClasses

--The indices of the classes used in this property
pullover = 2
coat = 4
sandal = 5
sneaker = 7


--Groups, tops and shoes. In group confusion is preferable to cross group confusion
tops : List Class
tops = [pullover, coat]

shoes : List Class
shoes = [sneaker, sandal]

--An Image is a tensor of rational numbers with dimensions 28x28
type Image = Tensor Rat [28, 28]

--Score is a rational number, it is the probability(not actually but logit) given to a class
--in the vector of class probabilities(logits)
type Score = Rat

--The network takes an image and returns a Vector of scores as long as the number of classes
@network
score : Image -> Vector Score numberOfClasses

--Vehicle works best(or only?) with normalised values so check that the pixels are all between 0 and 1
validPixel : Rat -> Bool
validPixel p = 0 <= p <= 1

--An image is valid if all the pixels in it are valid
validImage : Image -> Bool
validImage img = forall i j . validPixel (img ! i ! j)

--Is a class the first choice for the class of an image
isFirstChoice : Image -> Class -> Bool
isFirstChoice img topClass =
    let scores = score img in
    forall class . class != topClass => scores ! topClass > scores ! class

--Is a class the second choice for the class of an image
isSecondChoice : Image -> Class -> Bool
isSecondChoice img secondClass =
    let scores = score img in
    exists topClass . (isFirstChoice img topClass) and
    (forall class . class != topClass and class != secondClass => scores ! secondClass > scores ! class)

--No cross group confusion; if top choice in group 1 then second choice not in group 2
--For all classes in each class list
noConfusionBetween : Image -> List Class -> List Class -> Bool
noConfusionBetween img classList1 classList2 =
    forall class1 in classList1 .
	forall class2 in classList2 .
            not (isFirstChoice img class1 and isSecondChoice img class2)

--Does not confuse shoes with tops
--If an image has its first chosen class as one then the other should not be the second chosen class
@property
doesNotConfuseShoesWithTops : Bool
doesNotConfuseShoesWithTops = forall img . validImage img => noConfusionBetween img shoes tops
