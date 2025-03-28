import numpy as np
from mpi4py import MPI


def get_info(
    comm,
    rank: int,
    mp_size: int,
    dp_size: int,
    fc_layer: str,
    in_dim: int,
    out_dim: int,
):
    """
    Prepare necessary information for later communications in forward and backward passes.

    Parameters
    ----------
    comm : Communicator
        The global MPI communicator.
    rank : int
        The global rank of the process.
    mp_size : int
        Model Parallel size.
    dp_size : int
        Data Parallel size.
    fc_layer : str
        Identifier for the fully-connected layer. It must be one of:
        'fc_q', 'fc_k', 'fc_v', or 'fc_o'.
        - For 'fc_q', 'fc_k', and 'fc_v', the partitioning is along the output dimension.
        - For 'fc_o', the partitioning is along the input dimension.
    in_dim : int
        Original input feature dimension.
    out_dim : int
        Original output feature dimension.

    Returns
    -------
    mp_idx : int
        Model parallel index (position within a data parallel replica).
    dp_idx : int
        Data parallel index (which replica this process belongs to).
    mp_comm : Communicator
        The model parallel communicator (all processes in one data parallel replica).
    dp_comm : Communicator
        The data parallel communicator (all processes holding the same weight shard).
    part_in_dim : int
        The partitioned input dimension for the FC layer.
    part_out_dim : int
        The partitioned output dimension for the FC layer.
    """
    #TODO: Your code here

    # Compute model parallel and data parallel indices
    mp_idx = rank % mp_size  # Position within a model parallel replica
    dp_idx = rank // mp_size  # Which replica this process belongs to

    # Create model parallel and data parallel communicators
    mp_comm = comm.Split(color=dp_idx, key=mp_idx)  # All processes in the same DP group
    dp_comm = comm.Split(color=mp_idx, key=dp_idx)  # All processes holding same weight shard

    # Determine partitioned dimensions
    if fc_layer in {"fc_q", "fc_k", "fc_v"}:
        part_in_dim = in_dim
        part_out_dim = out_dim // mp_size  # Split across output dimension
    elif fc_layer == "fc_o":
        part_in_dim = in_dim // mp_size  # Split across input dimension
        part_out_dim = out_dim
    else:
        raise ValueError(f"Invalid fc_layer: {fc_layer}. Must be 'fc_q', 'fc_k', 'fc_v', or 'fc_o'.")

    return mp_idx, dp_idx, mp_comm, dp_comm, part_in_dim, part_out_dim


def naive_collect_forward_input(
    x: np.ndarray,
    mp_comm,
    mp_size: int,
):
    """
    Collects the fc_o layer's forward inputs from all model-parallel nodes.

    Each node holds a piece of the full input with shape:
      (batch_size, seq_length, part_in_dim)
    After gathering, the full input should have shape:
      (batch_size, seq_length, part_in_dim * mp_size)
    """
    batch_size, seq_length, part_in_dim = x.shape
    rank = mp_comm.Get_rank()

    # declare the buffer to gather the input from all model-parallel nodes
    collected_x = np.zeros((batch_size, seq_length, part_in_dim * mp_size), dtype=np.float64)

    # Ensure the input tensor is contiguous
    x = np.ascontiguousarray(x)

    # Use Sendrecv for better efficiency 
    for i in range(mp_size):
        if i != rank:
            send_buf = x.copy()  # Copy to ensure contiguous memory
            recv_buf = np.empty_like(x)

            mp_comm.Sendrecv(send_buf, dest=i, sendtag=99, recvbuf=recv_buf, source=i, recvtag=99)

            collected_x[:, :, i * part_in_dim : (i + 1) * part_in_dim] = recv_buf
        else:
            collected_x[:, :, i * part_in_dim : (i + 1) * part_in_dim] = x

    # Ensure the final result is contiguous before returning
    return np.ascontiguousarray(collected_x)


def naive_collect_forward_output(
    out: np.ndarray,
    mp_comm,
    mp_size: int,
):
    """
    Collects the fc_o layer's forward outputs from all model-parallel nodes.

    Each node holds a piece of the full output with shape:
      (batch_size, seq_length, part_out_dim)
    After gathering, the full output should have shape:
      (batch_size, seq_length, part_out_dim * mp_size)
    """
    #TODO: Your code here
    # batch_size, seq_length, part_out_dim = out.shape

    # # Allocate a buffer to gather the output from all model-parallel nodes
    # collected_out = np.empty((batch_size, seq_length, part_out_dim * mp_size), dtype=out.dtype)

    # # Use MPI Allgather to collect the output from all ranks
    # mp_comm.Allgather(out, collected_out)

    # # After Allgather, all processes will have the full output tensor
    # return collected_out

    batch_size, seq_length, part_out_dim = out.shape
    rank = mp_comm.Get_rank()

    # Allocate a buffer to gather the output from all model-parallel nodes
    collected_out = np.zeros((batch_size, seq_length, part_out_dim * mp_size), dtype=np.float64)

    # Ensure the output tensor is contiguous
    out = np.ascontiguousarray(out)

    # Use Sendrecv for better efficiency
    for i in range(mp_size):
        if i != rank:
            send_buf = out.copy()  # Copy to ensure contiguous memory
            recv_buf = np.empty_like(out)

            mp_comm.Sendrecv(send_buf, dest=i, sendtag=99, recvbuf=recv_buf, source=i, recvtag=99)

            collected_out[:, :, i * part_out_dim : (i + 1) * part_out_dim] = recv_buf
        else:
            collected_out[:, :, i * part_out_dim : (i + 1) * part_out_dim] = out

    # Ensure the final result is contiguous before returning
    return np.ascontiguousarray(collected_out)

def naive_collect_backward_output(
    output_grad: np.ndarray,
    mp_group_idx: int,
    mp_size: int,
):
    """
    Collect the fc output layer's output gradient for the local MP node.
    
    In our setup, the full output_grad is a 3-D tensor of shape 
        (batch_size, seq_length, out_dim),
    and the fully connected layer's weight is partitioned along out_dim.
    Therefore, we split output_grad along axis=2 into mp_size parts and
    return the part corresponding to mp_group_idx.
    
    Parameters
    ----------
    output_grad : np.ndarray
        The full output gradient from fc_o with shape 
        (batch_size, seq_length, out_dim).
    mp_group_idx : int
        The current model parallel node's index.
    mp_size : int
        The total number of model parallel nodes.
    
    Returns
    -------
    collected_output_grad : np.ndarray
        The local output gradient for this MP node with shape 
        (batch_size, seq_length, out_dim // mp_size).
    """
    #TODO: Your code here
    batch_size, seq_length, out_dim = output_grad.shape

    # Ensure out_dim is divisible by mp_size
    assert out_dim % mp_size == 0, "Output dimension must be divisible by model parallel size."

    # Compute the start and end indices for this node
    part_size = out_dim // mp_size
    start_idx = mp_group_idx * part_size
    end_idx = start_idx + part_size

    # Extract the relevant slice
    collected_output_grad = output_grad[:, :, start_idx:end_idx]

    print(730)
    print(collected_output_grad)

    return collected_output_grad


def naive_collect_backward_x(
    grad_x: np.ndarray,
    mp_comm,
    mp_size: int,
):
    """
    Use reduce-scatter / all-to-all to combine the contributions for grad_x from all nodes
    and scatter the reduced result along the input feature dimension.
    
    The grad_x tensor (gradient with respect to fc_o's input) has shape
        (batch_size, seq_length, in_dim),
    and the fc_o's weight matrix is sharded along the in_dim axis. In the 
    backward pass, each node computes a local grad_x and then these must be 
    summed across nodes. Instead of summing the full tensor and then slicing,
    we perform a reduce-scatter / all-to-all.
    
    Parameters
    ----------
    grad_x : np.ndarray
        The locally computed grad_x for fc_o, of shape 
        (batch_size, seq_length, in_dim).
    mp_comm :
        The model parallel communicator. It is assumed to expose methods such as reduce-scatter / all-to-all.
    mp_size : int
        The total number of model parallel nodes.
    
    Returns
    -------
    collected_grad_x : np.ndarray
        The reduced and scattered grad_x with shape 
        (batch_size, seq_length, in_dim // mp_size).
    """
    batch_size, seq_length, in_dim = grad_x.shape
    rank = mp_comm.Get_rank()

    # Ensure in_dim is divisible by mp_size
    assert in_dim % mp_size == 0, "Input dimension must be divisible by model parallel size."

    # Compute the partition size
    part_size = in_dim // mp_size

    # Ensure the grad_x tensor is contiguous
    grad_x = np.ascontiguousarray(grad_x)

    # Allocate a buffer for receiving the reduced result
    reduced_grad_x = np.zeros_like(grad_x)

    # Reduce: Sum all local grad_x values across nodes
    mp_comm.Allreduce(grad_x, reduced_grad_x, op=MPI.SUM)

    # Split the reduced_grad_x along columns (the last dimension)
    collected_grad_x = reduced_grad_x[:, :, rank * part_size : (rank + 1) * part_size]

    return collected_grad_x