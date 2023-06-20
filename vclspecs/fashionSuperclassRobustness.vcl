-- Inputs and outputs definitions
type Image = Tensor Rat [28, 28]

type Label = Index 10

--Labels
tshirt = 0
trouser = 1
pullover = 2
dress = 3
coat = 4
sandal = 5
shirt = 6
sneaker = 7
bag = 8
ankleBoot = 9

--Check that all Rat's in image are between 0 and 1
validImage : Image -> Bool
validImage x = forall i j . 0 <= x ! i ! j <= 1

@network
classifier : Image -> Vector Rat 10

--superclass definitions:
type Superclass = List Label

tops : Superclass
tops = [tshirt, pullover, dress, coat, shirt]

shoes : Superclass
shoes = [sandal, sneaker, ankleBoot]

bottoms : Superclass
bottoms = [trouser]

accessories : Superclass
accessories = [bag]

superclasses : List Superclass
superclasses = [tops, shoes, bottoms, accessories]

--Is a label in a given superclass
inSuperclass : Label -> List Label -> Bool
inSuperclass label superclass = exists class in superclass . label == class

--Which superclass is a label in
--whichSuperclass : Label -> Superclass
--whichSuperClass label = super in superclasses . inSuperclass label super

--x is an image i is a label. returns true when the classifier fits i label to x image,
--false when another label has a higher score.
advises : Image -> Label -> Bool
advises image label = forall j . j != label => classifier image ! label > classifier image ! j

advisesSuperclass : Image -> Label -> Bool
advisesSuperclass perturbedImage label =
    --false if : the perturbed image is advised as a label in a different superclass
    --true if : the perturbed image is advised as a label in the same superclass
    exists super in superclasses . 
        exists class in super . 
            inSuperclass label super and advises perturbedImage class

--epsilon ball around the image
--epsilon parameter passed at runtime
@parameter
epsilon : Rat

--defines x as an image where all pixels have values between positive and negative epsilon
boundedByEpsilon : Image -> Bool
boundedByEpsilon x = forall i j . -epsilon <= x ! i ! j <= epsilon

--Checks an images robustness to perturbations around a label
--Checking all perturbations means only need to check image - perturbation as 
--every possible perturbation is covered positive and negative
robustAround : Image -> Label -> Bool
robustAround image label = forall perturbation .
	let perturbedImage = image - perturbation in
	boundedByEpsilon perturbation and validImage perturbedImage =>
		advisesSuperclass perturbedImage label

--dataset size
@parameter(infer=True)
n : Nat

@dataset
trainingImages : Vector Image n

@dataset
trainingLabels : Vector Label n

@property
robust : Vector Bool n
robust = foreach i . robustAround (trainingImages ! i) (trainingLabels ! i)