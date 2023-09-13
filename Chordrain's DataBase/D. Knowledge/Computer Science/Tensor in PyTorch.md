#cs #deepLearning #python #pytorch 

# Tensor

你可以简单理解为，tensor 就是数据的高维表示。0 维 tensor 就是 scalar (标量)，1 维 tensor 就是 vector (向量)，2 维 tensor 就是 matrix (矩阵)。

## 1 创建

Tensors 与 Numpy 中的 ndarrays 类似，但是在 PyTorch 中 Tensors 可以使用 GPU 进行计算.

### 1.1 empty

创建一个 5x3 矩阵, 但是未初始化:

```python
x = torch.empty(5, 3)
print(x)
```

```
tensor([[0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000]])
```

### 1.2 rand

创建一个随机初始化的矩阵:

```python
x = torch.rand(5, 3)
print(x)
```

```
tensor([[0.6972, 0.0231, 0.3087],
        [0.2083, 0.6141, 0.6896],
        [0.7228, 0.9715, 0.5304],
        [0.7727, 0.1621, 0.9777],
        [0.6526, 0.6170, 0.2605]])
```

### 1.3 zeros

创建一个 0 填充的矩阵，数据类型为 long:

```python
x = torch.zeros(5, 3, dtype=torch.long)
print(x)
```

```
tensor([[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]])
```

### 1.4 使用现有数据

创建 tensor 并使用现有数据初始化:

```python
x = torch.tensor([5.5, 3])
print(x)
```

```
tensor([5.5000, 3.0000])
```

根据现有的张量创建张量。这些方法将重用输入张量的属性，例如， dtype，除非设置新的值进行覆盖。

```python
x = x.new_ones(5, 3, dtype=torch.double)      # new_* 方法来创建对象
print(x)

x = torch.randn_like(x, dtype=torch.float)    # 覆盖 dtype!
print(x)                                      # 对象的 size 是相同的，只是值和类型发生了变化
```

```
tensor([[1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.]], dtype=torch.float64)
tensor([[ 0.5691, -2.0126, -0.4064],
        [-0.0863,  0.4692, -1.1209],
        [-1.1177, -0.5764, -0.5363],
        [-0.4390,  0.6688,  0.0889],
        [ 1.3334, -1.1600,  1.8457]])
```

## 2 运算

### 2.1 基本运算

```python
ones = torch.ones(2, 3)
print(ones)

twos = torch.ones(2, 3) * 2 # every element is multiplied by 2
print(twos)

threes = ones + twos       # addition allowed because shapes are similar
print(threes)              # tensors are added element-wise
print(threes.shape)        # this has the same dimensions as input tensors

r1 = torch.rand(2, 3)
r2 = torch.rand(3, 2)
# uncomment this line to get a runtime error
# r3 = r1 + r2
```

```
tensor([[1., 1., 1.],
        [1., 1., 1.]])
tensor([[2., 2., 2.],
        [2., 2., 2.]])
tensor([[3., 3., 3.],
        [3., 3., 3.]])
torch.Size([2, 3])
```

### 2.2 abs

`abs` 函数对张量内部的值取绝对值:

```python
r = (torch.rand(2, 2) - 0.5) * 2 # values between -1 and 1
print('A random matrix, r:')
print(r)

# Common mathematical operations are supported:
print('\nAbsolute value of r:')
print(torch.abs(r))
```

```
A random matrix, r:
tensor([[ 0.9956, -0.2232],
        [ 0.3858, -0.6593]])

Absolute value of r:
tensor([[0.9956, 0.2232],
        [0.3858, 0.6593]])
```

### 2.3 三角函数

PyTorch 支持三角函数运算，例如 `sin`，`cos`，`tan`，`asine`，`acos`，`atan` ……