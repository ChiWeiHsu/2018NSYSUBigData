
# coding: utf-8

# In[1]:


def multiplication_table(m, n):
    for i in range(1, 10):
        for j in range(m, n+1):
            k=j*i
            print('%d X %d =%d\t'%(j,i,k),end='')
        print()
    print()


# In[2]:


def pyramid(n):
    for i in range(n):
        print(' '*(n-i-1)+'*'*(2*i-1))
