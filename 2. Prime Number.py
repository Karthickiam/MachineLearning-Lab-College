n = int(input("Enter the number: "))
flag = False
if n>1:
    for i in range(2,n):
        if(n%i)==0:
            flag = True
            break
if flag:
    print(n,"is not a Prime Number")
    print(i,"times",n//I,"is",n)
else:
    print(n,"is a Prime Number")
