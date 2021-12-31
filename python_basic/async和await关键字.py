"""
@Time : 2021/4/21 14:30
@Author : wmingzhu
@Annotation : 
"""
def func():
    print("test")

f1 = func()
print(f1)#执行到这里会输出test None。这是普通函数

#加上了async关键字以后就不再是普通函数，而是协程对象
async def func2():
    print("test2")
    print("test3")
    print("test4")

async def func3():
    print("test5")
    print("test6")
    print("test7")

f2 = func2()#这里并没有执行func2里面的print语句
f3 = func3()#同上

#类似与生成器的send方法，协程也有send方法。又类似于迭代器用raise StopIteration来发出迭代终止信号，协程也用StopIteration标志其结束
try:
    f2.send(None)
except StopIteration as e:
    pass

try:
    f3.send(None)
except StopIteration as e:
    pass

#如上协程对象正确执行。但是并没有体现出协同
#下面基于生成器，使用协程调用send方法，实现不同协程对象的交替执行

import types
@types.coroutine
def test():
    yield

async def t1():
    print("t1")
    await test()
    print("t1")
    await test()
    print("t1")
async def t2():
    print("t2")
    await test()
    print("t2")
    await test()
    print("t2")
    await test()
    print("t2")
async def t3():
    print("t3")
    await test()
    print("t3")
    await test()
    print("t3")
    await test()
    print("t3")
    await test()
    print("t3")

t1 = t1()
t2 = t2()
t3 = t3()
def do_by_turn(lis:list):
    while lis:
        for t in lis:
            try:
                t.send(None)
            except StopIteration as e:
                lis.remove(t)

do_by_turn([t1,t2,t3])

#==========================================================================================
#尝试相互等待
async def p1():
    print("1")
    print("2")
async def p2():
    print("3")
    await p1()
    print("4")

p2().send(None)
p2().send(None)
#输出结果是3，1，2，4







