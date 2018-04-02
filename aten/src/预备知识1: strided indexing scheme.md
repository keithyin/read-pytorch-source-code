##  数组步长 ##
首先介绍一下步长的概念即：相邻数组元素在内存中的开始地址的距离。数组步长如果等于数组元素的尺寸，则数组在内存中是连续的。比如int[10]，如果它是连续的，则第0个元素和第1个元素在内存中开始地址的距离为sizeof(int)，即刚好差一个元素的大小，如果不连续，则会大于一个元素的大小，因为它们两个之间多余字节。

## strided indexing scheme ##
在numpy、TH、opencv库中都会用到strided indexing scheme，在这些库中，有很多变量共享一块数据，只是“视图”不同。这里以numpy中的ndarray为例，多个数组变量共享数据，改动其中一个变量内的值，其他变量也会受影响。

```python
x = np.array([[1, 2, 3], [4, 5, 6]], np.int32)
y = x[:,1]
y[0] = 9
>>> y
array([9, 5])
>>> x
array([[1, 9, 3],
       [4, 5, 6]])
```
那它底层的是如何存储ndarray呢，首先ndarray在内存中实际上都作为一个内存块进行存储，在C语言看来它是一个一维数组或者说是由`malloc`或者`calloc`分配的某个给定大小的内存块，例如下表是一个有20个浮点类型（双精度）的内存块，它可能存储了一个4x5矩阵的值，也有可能存储了一个2x5x2的三阶张量的值。
![这里写图片描述](//img-blog.csdn.net/2018031921100225?watermark/2/text/Ly9ibG9nLmNzZG4ubmV0L3UwMTMwMTA4ODk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
有很多策略把ndarray的元素放到一个一维的内存块中，我们这里介绍strided indexing scheme，在访问时需要计算每个维度的步长。比如Fortran和 Matlab里都是列优先(按列存储元素)，一个shape为(2,3,4)的3维数组，第0维连续两个元素的起始位置相距1个itemsize(代入公式可得)，即[0,0,0]与[1,0,0]的起始位置就相差一个itemsize。但是如果在C中是行优先的，第0维连续两个元素的起始位置相距d1*d2=12个itemsize，即[0,0,0]与[1,0,0]的起始位置就相差12个itemsize
**itemsize代表每个元素的所占内存大小**
**如果该ndarray的所有元素在内存中都是连续的，则它的步长计算公式如下(column代表列优先, row代表行优先)**
![这里写图片描述](//img-blog.csdn.net/20180319213135991?watermark/2/text/Ly9ibG9nLmNzZG4ubmV0L3UwMTMwMTA4ODk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

```python
# storageOffset内存块的起始地址 offset为数组元素(i,j,k)的起始地址
offset = storageOffset + i * stride[0] + j * stride[1] + k * stride[2]
# 行优先和列优先介绍
[[1, 2, 3],
 [4, 5, 6]]
# 行优先
[1, 2, 3, 4, 5, 6]
# 列优先
[1, 4, 2, 5, 3, 6]
```
3维索引(i, j, k)的扩展: 一个N维索引(n1,n2,n3..nN-1)的偏移：每个维度的步长乘以该维度大小的和
![这里写图片描述](//img-blog.csdn.net/20180319214213714?watermark/2/text/Ly9ibG9nLmNzZG4ubmV0L3UwMTMwMTA4ODk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
实际中还有一点小情况
1. 某个维度k的size是1，这个维度的索引只能0，即nk=0，则该维度的步长可为任意，因为nk*sk恒等于0了
2. 某个数组没有任何元素，不存在合法的索引了，也就不要步长了。

## 检测内存是否连续 ##
有了上述步长的计算公式，在numpy或者TH低层库中check这个N维向量是否是内存连续的就可以根据stride和shape的公式进行检查，如果满足公式，就是连续的。
```c++
// 以TH库举例: aten/src/TH/generic/THTensor.cpp
int THTensor_(isContiguous)(const THTensor *self)
{
  int64_t z = 1;
  int d;
  for(d = self->nDimension-1; d >= 0; d--)
  {
    if(self->size[d] != 1)
    {
      if(self->stride[d] == z)
        z *= self->size[d];
      else
        return 0;
    }
  }
  return 1;
}
```

## 举例 ##

###  内存连续，视图不同 ##
这样当你改变ndarray视图时，就会产生不同的stride，比如(2,3,4)的3维数组，你改成了reshape成了(2,4,3)
本来的stride分别为：12, 4, 1 位置[1,2,2]  offset: 1\*12+2\*4+2=22  该位置元素为22
改为了:12, 3, 1 位置[1,2,2] offset: 1\*12+2\*3+2=20  该位置元素为20
低层还是这24个元素，但是各个视图由于stride的不同而不同

```python

a = np.arange((24))
# 假设这就是连续的内存块
array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22, 23])

b = np.reshape(a, (2,3,4))
array([[[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]],
       [[12, 13, 14, 15],
        [16, 17, 18, 19],
        [20, 21, 22, 23]]])
c = np.reshape(b, (2,4,3))
array([[[ 0,  1,  2],
        [ 3,  4,  5],
        [ 6,  7,  8],
        [ 9, 10, 11]],
       [[12, 13, 14],
        [15, 16, 17],
        [18, 19, 20],
        [21, 22, 23]]])


```

### 内存不连续 ###

举个2维的例子吧
```python
a: (3,4)
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
b = a[1:3,1:3]
array([[ 5,  6],
       [ 9, 10]])
c = a[:2,:2]
array([[0, 1],
       [4, 5]])

'''
b、c和a共占一块内存，但是b内的元素在内存中是不连续的，比如6和9; c中元素也不连续，比如1和4
a的 stride: (4, 1)  shape: (3,4) storageOffset=xx
b的 stride: (4, 1) shape: (2,2) 但是 storageOffset=xx+5
如果b是连续的它的stride和shape应该满足上文介绍的公式，stride应该为(2,1)
c的 stride: (4, 1)  shape: (2,2) storageOffset=xx
如果c是连续的它的stride和shape应该满足上文介绍的公式，stride应该为(2,1)
'''

```
### 思考：特别情况 ###
numpy中什么时候不共享内存呢？

```python
a = np.array([[1,-2],[-1,3]])
array([[ 1, -2],
       [-1,  3]])
b = a[a>0]
array([1,  3])
b[0] = 10
array([10,  3])
a # a内的元素不受影响，证明a和b已经不共享内存了
array([[ 1, -2],
       [-1,  3]])
```
上述例子的原因是什么呢？因为通过这种方式得到的b，已经不能仅通过改变视图即只通过stride和storageOffset，在a的内存上查看了。
如果不论怎样变化都共享内存的话，需要存储每个元素在原内存的位置，既不经济也不利于存储加速。
共享内存的情况：
1. 元素数量和顺序不变时，stride和shape可以变
2. 元素数量发生改变时，shape可以变，但是stride不能变了(逻辑内存是对齐的)


----------
[wiki：数组步长](https://zh.wikipedia.org/wiki/%E6%95%B0%E7%BB%84%E6%AD%A5%E9%95%BF)
[PyTorch源码浅析（一）](https://zhuanlan.zhihu.com/p/34496542)
[Internal memory layout of an ndarray](https://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html#internal-memory-layout-of-an-ndarray)