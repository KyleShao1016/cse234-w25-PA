from typing import Any, Dict, List
import torch
from auto_diff import *

class MatMulLayerNormOp(Op):
    """Fused matrix multiplication and layer normalization operation."""

    def __call__(
        self, 
        node_A: Node, 
        node_B: Node, 
        normalized_shape: List[int], 
        eps: float = 1e-5
    ) -> Node:
        """
        Args:
            node_A: The first input node.
            node_B: The second input node.
            normalized_shape: The shape of the normalization axes.
            eps: The epsilon value to avoid division by zero.
        """
        return Node(
            inputs=[node_A, node_B],
            op=self,
            attrs={
                "normalized_shape": normalized_shape,
                "eps": eps
            },
            name=f"MatMulLayerNorm({node_A.name}@{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the fused matmul and layer normalization result."""
        assert len(input_values) == 2
        """TODO: your code here"""
        # matmul_res = input_values[0] @ input_values[1]

        matmul_res = torch.matmul(input_values[0], input_values[1])
        eps = node.attrs['eps']

        mean = matmul_res.mean(dim=-1, keepdim=True)
        var = matmul_res.var(dim=-1, keepdim=True, unbiased=False)

        norm = (matmul_res - mean) / torch.sqrt(var + eps)
        return norm
        
        # raise NotImplementedError

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of fused node, return partial adjoints to each input."""
        """TODO: your code here"""
        # Retrieve input nodes
        input1, input2 = node.inputs
        eps = node.attrs["eps"]
        n = node.attrs['normalized_shape'][-1]

        # Forward computation for mean and variance (recompute for gradient)
        matmul_res = matmul(input1, input2)
        _mean = mean(matmul_res, dim=(-1), keepdim=True)

        # calculate the standard deviation
        diff = sub(matmul_res, _mean)
        squared_diff = power(diff, 2)
        _var = mean(squared_diff, dim=(-1), keepdim=True)

        std_inv = power(sqrt(add_by_const(_var, eps)), -1)

        normalized_input = mul(diff, std_inv)
        dL_dNorm = output_grad

        dL_dMean = mul(sum_op(dL_dNorm, dim=(-1), keepdim=True), mul_by_const(std_inv, -1))

        # dL_dMean = dL_dNorm.sum(dim=-1, keepdim=True) * -std_inv
        s = mul(dL_dNorm, normalized_input)
        t = power(std_inv, 3)
        dL_dVar = sum_op(mul(mul_by_const(s, -0.5), t), dim=(-1), keepdim=True)
        
        u = mul_by_const(dL_dVar, 2)
        dL_dNorm = add(add(mul(dL_dNorm, std_inv), div_by_const(dL_dMean, n)), div_by_const(mul(u, normalized_input), n))

        grad1 = matmul(dL_dNorm, transpose(input2, -1, -2))  
        grad2 = matmul(transpose(input1, -1, -2), dL_dNorm) 
        return [grad1, grad2]

        # raise NotImplementedError


class MatMulSoftmaxOp(Op):
    """Fused matrix multiplication and softmax operation."""

    def __call__(
        self, 
        node_A: Node, 
        node_B: Node, 
        dim: int = -1
    ) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            attrs={
                "dim": dim
            },
            name=f"MatMulSoftmax({node_A.name}@{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the fused matmul and softmax result."""
        assert len(input_values) == 2
        """TODO: your code here"""
        dim = node.attrs["dim"]
        # matmul_res = input_values[0] @ input_values[1]
        matmul_res = torch.matmul(input_values[0], input_values[1])
        # Apply numerical stability trick: subtract max before exponentiation
        matmul_res_stable = matmul_res - matmul_res.max(dim=dim, keepdim=True)[0]
        # print(matmul_res_stable)
        # Compute softmax
        exp_tensor = torch.exp(matmul_res_stable)
        # print(exp_tensor)
        sum_exp = torch.sum(exp_tensor, dim=dim, keepdim=True)
        # print(sum_exp)
        softmax_res = exp_tensor / sum_exp
        # softmax_res = torch.div(exp_tensor, sum_exp)

        return softmax_res
        # raise NotImplementedError

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of fused node, return partial adjoints to each input."""
        # First compute the forward pass result we need for softmax gradient
        """TODO: your code here"""
        dim = node.attrs["dim"]
        input1, input2 = node.inputs

        matmul_res = matmul(input1, input2)
        softmax_output = softmax(matmul_res, dim=dim)

        # Compute gradient of softmax w.r.t. matmul_res
        s = mul(output_grad, softmax_output)
        t = sub(output_grad, sum_op(s, dim=dim, keepdim=True))
        grad_wrt_matmul = mul(softmax_output, t)

        grad_A = matmul(grad_wrt_matmul, transpose(input2, -1, -2))
        grad_B = matmul(transpose(input1, -1, -2), grad_wrt_matmul)

        return [grad_A, grad_B]

        # raise NotImplementedError

# Create global instances of the fused ops
matmul_layernorm = MatMulLayerNormOp()
matmul_softmax = MatMulSoftmaxOp()