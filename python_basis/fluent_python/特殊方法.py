#特殊方法即类中前后都带有双下划线的方法，又叫魔术方法magic或者双下方法dunder,特殊方法是解释器自动调用，而不是代码显式调用
import collections
from random import choice

#使用namedtuple创建一个简单的类，它只有属性而没有方法
Card = collections.namedtuple("Card",['rank','suit'])
class Cards:
    ranks = [str(n) for n in range(2,11)] + list('JQKA')
    suits = 'spades diamonds clubs hearts'.split()

    def __init__(self):
        self._cards = [Card(rank,suit) for rank in self.ranks for suit in self.suits]

    def __len__(self):
        return len(self._cards)

    def __getitem__(self, item):
        return self._cards[item]

#索引
cards = Cards()
print(choice(cards))#choice函数会随机返回序列的一个元素
#切片
print(cards[:3])
print(cards[3::12])#起始:结束:步长

#迭代
print("-----------------------------")
for card in cards:
    print(card)
print("-----------------------------")
#反迭代
print("-----------------------------")
for card in reversed(cards):
    print(card)
print("-----------------------------")
#上面操作的本质都是把item索引值传给__getitem__函数