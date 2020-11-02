

x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
x_iter = iter(x)

l = list([])

index = 0

for j in range(0, len(x)):

    temp = list([])

    for i in range(0, len(x)):

        if j == i:
            temp.append(0)
            
            next_ = next(x_iter, None)
            continue

        curr = next_
        next_ = next(x_iter, None)

        if next_ == None:
            temp.append(0)
            break

        temp.append(next_-x[j])
        index += 1

   
    l.append(temp)
    x_iter = iter(x)

print(l)
