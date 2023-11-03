import os
import sys
import numpy as np
import einops
from typing import Union, Optional, Tuple
import torch as t
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
import functools
import math
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from tqdm.notebook import tqdm
from collections import namedtuple

# Make sure exercises are in the path
section_dir = Path(__file__).parent
exercises_dir = section_dir.parent
assert exercises_dir.name == "exercises", f"This file should be run inside 'exercises/part2_cnns', not '{section_dir}'"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from plotly_utils import imshow, line, bar
from part2_cnns.utils import display_array_as_img, display_soln_array_as_img
import part2_cnns.tests as tests

MAIN = __name__ == "__main__"
# Your code here - define arr1

arr = np.load(section_dir / "numbers.npy")

arr1 = einops.rearrange(arr, 'b c h w -> c h (b w)')

# if MAIN:
    # display_array_as_img(arr1)

arr2 = einops.repeat(arr[0], 'c h w -> c (2 h) w')

# if MAIN:
#     display_array_as_img(arr2)

# first2 = einops.rearrange([arr[0], arr[1]], 'b c h w -> c h (b w)')
# arr3 = einops.repeat(first2, 'c h w -> c (2 h) w')
arr3 = einops.repeat(arr[0:2], 'b c h w -> c (b h) (2 w)')

# if MAIN:
#     display_array_as_img(arr3)

arr4 = einops.repeat(arr[0], 'c h w -> c (h 2) w')

# if MAIN:
#     display_array_as_img(arr4)

arr5 = einops.reduce(arr[0], 'c h w -> h (c w)', 'mean')

# if MAIN:
#     display_array_as_img(arr5)

arr6 = einops.rearrange(arr, '(b1 b2) c h w -> c (b1 h) (b2 w) ', b1=2)

# if MAIN:
#     display_array_as_img(arr6)

# Your code here - define arr7

# blackImgs = einops.reduce(arr, 'b c h w -> b c () ()', 'max') - arr
# blackImgs /= einops.reduce(blackImgs, 'b c h w -> b c () ()', 'max')
# arr7 = einops.rearrange(blackImgs, 'b c h w -> c h (b w)')
arr7 = einops.reduce(arr.astype(float), "b c h w -> h (b w)", "max").astype(int)

# if MAIN:
#     display_array_as_img(arr7)


# Your code here - define arr8
arr8 = einops.reduce(arr, 'b c h w -> h w', 'min')

# if MAIN:
#     display_array_as_img(arr8)

arr9 = einops.rearrange(arr[1], 'c h w -> c w h')
# if MAIN:
#     display_array_as_img(arr9)


# arr10 = einops.reduce(arr6.astype(float), 'c (h h2) (w w2) -> c h w', 'mean', h2=2, w2=2).astype(int)
arr10 = einops.reduce(arr, "(b1 b2) c (h h2) (w w2) -> c (b1 h) (b2 w)", "max", h2=2, w2=2, b1=2)
# if MAIN:
#     display_array_as_img(arr10)


def einsum_trace(mat: np.ndarray):
    '''
    Returns the same as `np.trace`.
    '''
    pass
    return einops.einsum(mat, "i i ->")

def einsum_mv(mat: np.ndarray, vec: np.ndarray):
    '''
    Returns the same as `np.matmul`, when `mat` is a 2D array and `vec` is 1D.
    '''
    pass
    return einops.einsum(mat, vec, "i j, j -> i")

def einsum_mm(mat1: np.ndarray, mat2: np.ndarray):
    '''
    Returns the same as `np.matmul`, when `mat1` and `mat2` are both 2D arrays.
    '''
    pass
    return einops.einsum(mat1, mat2, "i j, j k -> i k")

def einsum_inner(vec1: np.ndarray, vec2: np.ndarray):
    '''
    Returns the same as `np.inner`.
    '''
    pass
    return einops.einsum(vec1, vec2, "i, i ->")

def einsum_outer(vec1: np.ndarray, vec2: np.ndarray):
    '''
    Returns the same as `np.outer`.
    '''
    pass
    return einops.einsum(vec1, vec2, "i, j -> i j")


if MAIN:
    tests.test_einsum_trace(einsum_trace)
    tests.test_einsum_mv(einsum_mv)
    tests.test_einsum_mm(einsum_mm)
    tests.test_einsum_inner(einsum_inner)
    tests.test_einsum_outer(einsum_outer)

if MAIN:
    test_input = t.tensor(
        [[0, 1, 2, 3, 4], 
        [5, 6, 7, 8, 9], 
        [10, 11, 12, 13, 14], 
        [15, 16, 17, 18, 19]], dtype=t.float
    )
if MAIN:
    TestCase = namedtuple("TestCase", ["output", "size", "stride"])

    test_cases = [
        TestCase(
            output=t.tensor([0, 1, 2, 3]), 
            size=(4,),
            stride=(1,),
        ),
        TestCase(
            output=t.tensor([[0, 2], [5, 7]]), 
            size=(2, 2),
            stride=(5, 2),
        ),

        TestCase(
            output=t.tensor([0, 1, 2, 3, 4]),
            size=(5,),
            stride=(1,),
        ),

        TestCase(
            output=t.tensor([0, 5, 10, 15]),
            size=(4, ),
            stride=(5, ),
        ),

        TestCase(
            output=t.tensor([
                [0, 1, 2], 
                [5, 6, 7]
            ]), 
            size=(2, 3),
            stride=(5, 1),
        ),

        TestCase(
            output=t.tensor([
                [0, 1, 2], 
                [10, 11, 12]
            ]), 
            size=(2, 3),
            stride=(10, 1),
        ),

        TestCase(
            output=t.tensor([
                [0, 0, 0], 
                [11, 11, 11]
            ]), 
            size=(2, 3),
            stride=(11, 0),
        ),

        TestCase(
            output=t.tensor([0, 6, 12, 18]), 
            size=(4,),
            stride=(6,),
        ),
    ]

    for (i, test_case) in enumerate(test_cases):
        if (test_case.size is None) or (test_case.stride is None):
            print(f"Test {i} failed: attempt missing.")
        else:
            actual = test_input.as_strided(size=test_case.size, stride=test_case.stride)
            if (test_case.output != actual).any():
                print(f"Test {i} failed:")
                print(f"Expected: {test_case.output}")
                print(f"Actual: {actual}\n")
            else:
                print(f"Test {i} passed!\n")


def as_strided_trace(mat: Float[Tensor, "i j"]) -> Float[Tensor, ""]:
    '''
    Returns the same as `torch.trace`, using only `as_strided` and `sum` methods.
    '''
    pass
    stride = mat.stride()
    strided = mat.as_strided((mat.size(0),), (stride[0] + stride[1],))
    # return einops.einsum(strided, "i ->")
    return strided.sum()


if MAIN:
    tests.test_trace(as_strided_trace)

def as_strided_mv(mat: Float[Tensor, "i j"], vec: Float[Tensor, "j"]) -> Float[Tensor, "i"]:
    '''
    Returns the same as `torch.matmul`, using only `as_strided` and `sum` methods.
    '''
    pass
    strideV = vec.stride()
    vec_expanded = vec.as_strided(mat.shape, (0, strideV[0]))
    merged = mat * vec_expanded
    return merged.sum(1)


if MAIN:
    tests.test_mv(as_strided_mv)
    tests.test_mv2(as_strided_mv)

def as_strided_mm(matA: Float[Tensor, "i j"], matB: Float[Tensor, "j k"]) -> Float[Tensor, "i k"]:
    '''
    Returns the same as `torch.matmul`, using only `as_strided` and `sum` methods.
    '''
    pass
    expandedSize = (matA.shape[0], matA.shape[1], matB.shape[1])
    expandedA = matA.as_strided(expandedSize, (matA.stride(0), matA.stride(1), 0))
    expandedB = matB.as_strided(expandedSize, (0, matB.stride(0), matB.stride(1)))
    output = expandedA * expandedB
    return output.sum(1)


if MAIN:
    tests.test_mm(as_strided_mm)
    tests.test_mm2(as_strided_mm)

def conv1d_minimal_simple(x: Float[Tensor, "w"], weights: Float[Tensor, "kw"]) -> Float[Tensor, "ow"]:
    '''
    Like torch's conv1d using bias=False and all other keyword arguments left at their default values.

    Simplifications: batch = input channels = output channels = 1.

    x: shape (width,)
    weights: shape (kernel_width,)

    Returns: shape (output_width,)
    '''
    pass
    kw = weights.shape[0]
    w = x.shape[0]
    out_width = w - kw + 1
    stride = x.stride(0)

    expandedX = x.as_strided((out_width, kw), (stride, stride))
    result = einops.einsum(expandedX, weights, "w kw, kw -> w")
    return result


if MAIN:
    tests.test_conv1d_minimal_simple(conv1d_minimal_simple)


def conv1d_minimal(x: Float[Tensor, "b ic w"], weights: Float[Tensor, "oc ic kw"]) -> Float[Tensor, "b oc ow"]:
    '''
    Like torch's conv1d using bias=False and all other keyword arguments left at their default values.

    x: shape (batch, in_channels, width)
    weights: shape (out_channels, in_channels, kernel_width)

    Returns: shape (batch, out_channels, output_width)
    '''
    pass

    (out_channels, in_channels, kernel_width) = weights.shape
    (batch, in_chanelsx, width) = x.shape
    out_width = width - kernel_width + 1
    (s_batch, s_in_channels, s_w) = x.stride()
    expandedX = x.as_strided((batch, in_channels, out_width, kernel_width), (s_batch, s_in_channels, s_w, s_w))
    result = einops.einsum(expandedX, weights, "batch in_channels out_width kernel_width, out_channels in_channels kernel_width -> batch out_channels out_width")
    return result


if MAIN:
    tests.test_conv1d_minimal(conv1d_minimal)


def conv2d_minimal(x: Float[Tensor, "b ic h w"], weights: Float[Tensor, "oc ic kh kw"]) -> Float[Tensor, "b oc oh ow"]:
    '''
    Like torch's conv2d using bias=False and all other keyword arguments left at their default values.

    x: shape (batch, in_channels, height, width)
    weights: shape (out_channels, in_channels, kernel_height, kernel_width)

    Returns: shape (batch, out_channels, output_height, output_width)
    '''
    pass
    (b, ic, h, w) = x.shape
    (oc, icw, kh, kw) = weights.shape
    (sb, sic, sh, sw) = x.stride()

    output_height = h - kh + 1
    output_width = w - kw + 1
    new_shape =  (b, ic, output_height, output_width, kh, kw)
    stride = (sb, sic, sh, sw, sh, sw)
    expandedX = x.as_strided(new_shape, stride)
    result = einops.einsum(expandedX, weights, "b ic oh ow kh kw, oc ic kh kw -> b oc oh ow")
    return result


if MAIN:
    tests.test_conv2d_minimal(conv2d_minimal)


def pad1d(x: t.Tensor, left: int, right: int, pad_value: float) -> t.Tensor:
    '''Return a new tensor with padding applied to the edges.

    x: shape (batch, in_channels, width), dtype float32

    Return: shape (batch, in_channels, left + right + width)
    '''
    pass
    size = (x.shape[0], x.shape[1], x.shape[2] + left + right)
    output = x.new_full(size, fill_value=pad_value)
    output[..., left :  x.shape[2] + left] = x
    return output


if MAIN:
    tests.test_pad1d(pad1d)
    tests.test_pad1d_multi_channel(pad1d)


def pad2d(x: t.Tensor, left: int, right: int, top: int, bottom: int, pad_value: float) -> t.Tensor:
    '''Return a new tensor with padding applied to the edges.

    x: shape (batch, in_channels, height, width), dtype float32

    Return: shape (batch, in_channels, top + height + bottom, left + width + right)
    '''
    pass
    b, i, h, w = x.shape
    size = (b, i , h + top + bottom, w + left + right)
    output = x.new_full(size, fill_value=pad_value)
    output[..., top : h + top, left :  w + left] = x

    return output


if MAIN:
    tests.test_pad2d(pad2d)
    tests.test_pad2d_multi_channel(pad2d)

def conv1d(
    x: Float[Tensor, "b ic w"], 
    weights: Float[Tensor, "oc ic kw"], 
    stride: int = 1, 
    padding: int = 0
) -> Float[Tensor, "b oc ow"]:
    '''
    Like torch's conv1d using bias=False.

    x: shape (batch, in_channels, width)
    weights: shape (out_channels, in_channels, kernel_width)

    Returns: shape (batch, out_channels, output_width)
    '''
    pass
    padded = pad1d(x, left=padding, right=padding, pad_value=0)

    b, ic, w = padded.shape
    oc, ic2, kw = weights.shape

    output_width = 1 + (w - kw) // stride
    sb, sic, sw = padded.stride()
    new_shape =  (b, ic, output_width, kw)
    stride_size = (sb, sic, sw * stride,  sw)
    expandedX = padded.as_strided(size=new_shape, stride=stride_size)

    return einops.einsum(expandedX, weights, "b ic ow kw, oc ic kw -> b oc ow")


if MAIN:
    tests.test_conv1d(conv1d)


IntOrPair = Union[int, Tuple[int, int]]
Pair = Tuple[int, int]

def force_pair(v: IntOrPair) -> Pair:
    '''Convert v to a pair of int, if it isn't already.'''
    if isinstance(v, tuple):
        if len(v) != 2:
            raise ValueError(v)
        return (int(v[0]), int(v[1]))
    elif isinstance(v, int):
        return (v, v)
    raise ValueError(v)

# Examples of how this function can be used:

def conv2d(
    x: Float[Tensor, "b ic h w"], 
    weights: Float[Tensor, "oc ic kh kw"], 
    stride: IntOrPair = 1, 
    padding: IntOrPair = 0
) -> Float[Tensor, "b oc oh ow"]:
    '''
    Like torch's conv2d using bias=False

    x: shape (batch, in_channels, height, width)
    weights: shape (out_channels, in_channels, kernel_height, kernel_width)

    Returns: shape (batch, out_channels, output_height, output_width)
    '''
    pass
    paddingY, paddingX = force_pair(padding)
    padded = pad2d(x, left=paddingX, right=paddingX, top=paddingY, bottom=paddingY, pad_value=0)

    b, ic, h, w = padded.shape
    oc, ic2, kh, kw = weights.shape

    strideY, strideX = force_pair(stride)
    ow = 1 + (w - kw) // strideX
    oh = 1 + (h - kh) // strideY
    sb, sic, sh, sw = padded.stride()
    new_shape =  (b, ic, oh, ow, kh, kw)
    stride_size = (sb, sic, sh * strideY, sw * strideX, sh, sw)
    strided = padded.as_strided(size=new_shape, stride=stride_size)

    return einops.einsum(strided, weights, "b ic oh ow kh kw, oc ic kh kw -> b oc oh ow")

if MAIN:
    tests.test_conv2d(conv2d)

def maxpool2d(
    x: Float[Tensor, "b ic h w"], 
    kernel_size: IntOrPair, 
    stride: Optional[IntOrPair] = None, 
    padding: IntOrPair = 0
) -> Float[Tensor, "b ic oh ow"]:
    '''
    Like PyTorch's maxpool2d.

    x: shape (batch, channels, height, width)
    stride: if None, should be equal to the kernel size

    Return: (batch, channels, output_height, output_width)
    '''
    pass
    paddingY, paddingX = force_pair(padding)
    padded = pad2d(x, left=paddingX, right=paddingX, top=paddingY, bottom=paddingY, pad_value=-t.inf)

    b, ic, h, w = padded.shape
    stride = stride if stride != None else kernel_size

    kh, kw = force_pair(kernel_size)
    strideY, strideX = force_pair(stride)
    oh = 1 + (h - kh) // strideY
    ow = 1 + (w - kw) // strideX
    sb, sic, sh, sw = padded.stride()
    new_shape =  (b, ic, oh, ow, kh, kw)
    stride_size = (sb, sic, sh * strideY, sw * strideX, sh, sw)
    strided = padded.as_strided(size=new_shape, stride=stride_size)

    return t.amax(strided, dim=[-1, -2])

if MAIN:
    tests.test_maxpool2d(maxpool2d)

class MaxPool2d(nn.Module):
    def __init__(self, kernel_size: IntOrPair, stride: Optional[IntOrPair] = None, padding: IntOrPair = 1):
        super().__init__()
        self.kh, self.kw = force_pair(kernel_size)
        unwrappedStride = stride if stride != None else kernel_size
        self.strideY, self.strideX = force_pair(unwrappedStride)
        self.paddingY, self.paddingX = force_pair(padding)

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Call the functional version of maxpool2d.'''
        padded = pad2d(x, left=self.paddingX, right=self.paddingX, top=self.paddingY, bottom=self.paddingY, pad_value=-t.inf)
        b, ic, h, w = padded.shape
        oh = 1 + (h - self.kh) // self.strideY
        ow = 1 + (w - self.kw) // self.strideX
        sb, sic, sh, sw = padded.stride()
        new_shape =  (b, ic, oh, ow, self.kh, self.kw)
        stride_size = (sb, sic, sh * self.strideY, sw * self.strideX, sh, sw)
        strided = padded.as_strided(size=new_shape, stride=stride_size)

        return t.amax(strided, dim=[-1, -2])

    def extra_repr(self) -> str:
        '''Add additional information to the string representation of this class.'''
        pass
        return "Maxpool2D paddingX = " + str(self.paddingX)


if MAIN:
    tests.test_maxpool2d_module(MaxPool2d)
    m = MaxPool2d(kernel_size=3, stride=2, padding=1)
    print(f"Manually verify that this is an informative repr: {m}")


class ReLU(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        x[x < 0] = 0
        return x


if MAIN:
    tests.test_relu(ReLU)

class Flatten(nn.Module):
    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input: t.Tensor) -> t.Tensor:
        '''
        Flatten out dimensions from start_dim to end_dim, inclusive of both.
        '''
        pass
        current_shape = input.shape
        end_dim = self.end_dim if self.end_dim >= 0 else len(current_shape) + self.end_dim

        shape_left = current_shape[:self.start_dim]
        # shape_middle = t.prod(t.tensor(shape[start_dim : end_dim+1])).item()
        shape_middle = functools.reduce(lambda x, y: x*y, current_shape[self.start_dim : end_dim+1])
        shape_right = current_shape[end_dim+1:]
        new_shape = shape_left + (shape_middle,) + shape_right
        output = t.reshape(input, shape=new_shape)
        return output

    def extra_repr(self) -> str:
        return ", ".join([f"{key}={getattr(self, key)}" for key in ["start_dim", "end_dim"]])



if MAIN:
    tests.test_flatten(Flatten)

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias=True):
        '''
        A simple linear (technically, affine) transformation.

        The fields should be named `weight` and `bias` for compatibility with PyTorch.
        If `bias` is False, set `self.bias` to None.
        '''
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        kaiming = 1 / math.sqrt(in_features)

        weight = kaiming * (2 * t.rand(out_features, in_features) - 1)
        self.weight = nn.Parameter(weight)
        if (bias):
            bias = kaiming * (2 * t.rand(out_features,) - 1)
            self.bias = nn.Parameter(bias)
        else:
            self.bias = None

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        x: shape (*, in_features)
        Return: shape (*, out_features)
        '''
        pass
        x = einops.einsum(x, self.weight, "... in, out in -> ... out")
        if self.bias is not None:
            x += self.bias
        return x

    def extra_repr(self) -> str:
        pass
        return ", ".join([f"{key}={getattr(self, key)}" for key in ["start_dim", "end_dim"]])


if MAIN:
    tests.test_linear_forward(Linear)
    tests.test_linear_parameters(Linear)
    tests.test_linear_no_bias(Linear)

class Conv2d(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: IntOrPair, stride: IntOrPair = 1, padding: IntOrPair = 0
    ):
        '''
        Same as torch.nn.Conv2d with bias=False.

        Name your weight field `self.weight` for compatibility with the PyTorch version.
        '''
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kh, self.kw = force_pair(kernel_size)
        self.strideY, self.strideX = force_pair(stride)
        self.paddingY, self.paddingX = force_pair(padding)

        sf = 1 / np.sqrt(in_channels * self.kh * self.kw)
        weight = sf * (2 * t.rand(out_channels, in_channels, self.kh, self.kw) - 1)
        self.weight = nn.Parameter(weight)

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Apply the functional conv2d you wrote earlier.'''
        padded = pad2d(x, left=self.paddingX, right=self.paddingX, top=self.paddingY, bottom=self.paddingY, pad_value=0)

        b, ic, h, w = padded.shape

        ow = 1 + (w - self.kw) // self.strideX
        oh = 1 + (h - self.kh) // self.strideY
        sb, sic, sh, sw = padded.stride()
        new_shape =  (b, ic, oh, ow, self.kh, self.kw)
        stride_size = (sb, sic, sh * self.strideY, sw * self.strideX, sh, sw)
        strided = padded.as_strided(size=new_shape, stride=stride_size)

        return einops.einsum(strided, self.weight, "b ic oh ow kh kw, oc ic kh kw -> b oc oh ow")

    def extra_repr(self) -> str:
        pass


if MAIN:
    tests.test_conv2d_module(Conv2d)