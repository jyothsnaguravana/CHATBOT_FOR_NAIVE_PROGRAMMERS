nums = list(map(int , input()))
n = len(nums)
temp = []
nums.reverse()
temp = nums
nums.reverse()  
l  = []
# print(temp) 
l.append(temp[0])
for i in range(1,len(temp)):
    l.append(temp[i] + l[i-1]) 
print(l)
ans = []
sum = ''
for i in range(n):
    if(nums[i] % 2 == 0):
        for j in range(i+1,n):
            if(l[j] % 2 == 1):
                sum += str(l[j])
    else:
        for j in range(i+1,n):
            if(l[j] % 2 == 0):
                sum += str(l[j])

# print(sum)




# print(l)
#9880127431
#26971

