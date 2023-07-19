--check use of <= in property

@property
monotonicIncrease : Bool
monotonicIncrease = 
    forall i .
        forall j .
            i <= j => regression i <= regression j

@property
monotonicDecrease : Bool
monotonicDecrease = 
    forall i .
        forall j .
            i > j => regression i > regression j