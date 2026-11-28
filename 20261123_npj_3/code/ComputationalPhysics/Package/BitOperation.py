import numpy as np

def SetBit(i, n):
    """
    set nth bit to 1
    """
    return i|(1<<n)

def ClearBit(i,n):
    """
    clear nth bit to 0
    """
    return i&~(1<<n)

def FlipBit(i,n):
    """
    Flip nth bit
    """
    return i^(1<<n)

def ReadBit(i, n):
    """
    read nth bit. 0 corresponds to the right most bit.
    """
    return (i&(1<<n))>>n

def PopCntBit(i):
    """
    count how many 1 bits
    """
    return bin(i).count("1")

def PickBit(i, k, n):
    """
    pick up nbits from kth bit
    """
    return (i&((2**n-1)<<k))>>k

def RotBit(i,L,n):
    """
    circular bit shift left
    """
    return (PickBit(i,0,L-n)<<n)+(i>>(L-n))

def JointBit(i, j, k):
    """
    joint two seperated bit into one bit, such as 1000+0001->10000001. k is the number of bit i shifted!!
    """
    return (i<<k) + j


if __name__=="__main__":
    print(1)