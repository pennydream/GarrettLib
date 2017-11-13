from abc import ABCMeta, abstractmethod

class wallet(object):
    __metaclass__ = ABCMeta 
    def getBalance():pass
    "returns the balance of wallet"
    def addBalance():pass
    "adds to the balance of wallet"
    def spendMoney():pass
    "Removes from the balance of the wallet"

class MyWallet(wallet):

    def __init__(self, balance):
        self.balance = balance

    def getBalance():
        return self.balance

    def addBalance(more):
        self.balance += more

    def spendMoney(less):
        self.balance -= less


if __name__ == "__main__":
    print "Hello World!"
