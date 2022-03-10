
def fibonacci(n:int):
    a = 1
    b = 1
    if type(n) != int or n < 0:
        return "请输入正整数"
    elif n ==1 or n ==2:
        return 1
    else:
        for i in range(3,n+1):
            c = a + b
            a = b
            b = c
        return c

print(fibonacci(6))

