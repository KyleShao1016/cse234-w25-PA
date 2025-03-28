from mpi4py import MPI
import numpy as np

class Communicator(object):
    def __init__(self, comm: MPI.Comm):
        self.comm = comm
        self.total_bytes_transferred = 0

    def Get_size(self):
        return self.comm.Get_size()

    def Get_rank(self):
        return self.comm.Get_rank()

    def Barrier(self):
        return self.comm.Barrier()

    def Allreduce(self, src_array, dest_array, op=MPI.SUM):
        assert src_array.size == dest_array.size
        src_array_byte = src_array.itemsize * src_array.size
        self.total_bytes_transferred += src_array_byte * 2 * (self.comm.Get_size() - 1)
        self.comm.Allreduce(src_array, dest_array, op)

    def Allgather(self, src_array, dest_array):
        src_array_byte = src_array.itemsize * src_array.size
        dest_array_byte = dest_array.itemsize * dest_array.size
        self.total_bytes_transferred += src_array_byte * (self.comm.Get_size() - 1)
        self.total_bytes_transferred += dest_array_byte * (self.comm.Get_size() - 1)
        self.comm.Allgather(src_array, dest_array)

    def Reduce_scatter(self, src_array, dest_array, op=MPI.SUM):
        src_array_byte = src_array.itemsize * src_array.size
        dest_array_byte = dest_array.itemsize * dest_array.size
        self.total_bytes_transferred += src_array_byte * (self.comm.Get_size() - 1)
        self.total_bytes_transferred += dest_array_byte * (self.comm.Get_size() - 1)
        self.comm.Reduce_scatter_block(src_array, dest_array, op)

    def Split(self, key, color):
        return __class__(self.comm.Split(key=key, color=color))

    def Alltoall(self, src_array, dest_array):
        nprocs = self.comm.Get_size()

        # Ensure that the arrays can be evenly partitioned among processes.
        assert src_array.size % nprocs == 0, (
            "src_array size must be divisible by the number of processes"
        )
        assert dest_array.size % nprocs == 0, (
            "dest_array size must be divisible by the number of processes"
        )

        # Calculate the number of bytes in one segment.
        send_seg_bytes = src_array.itemsize * (src_array.size // nprocs)
        recv_seg_bytes = dest_array.itemsize * (dest_array.size // nprocs)

        # Each process sends one segment to every other process (nprocs - 1)
        # and receives one segment from each.
        self.total_bytes_transferred += send_seg_bytes * (nprocs - 1)
        self.total_bytes_transferred += recv_seg_bytes * (nprocs - 1)

        self.comm.Alltoall(src_array, dest_array)

    def myAllreduce(self, src_array, dest_array, op=MPI.SUM):
        """
        A manual implementation of all-reduce using a reduce-to-root
        followed by a broadcast.
        
        Each non-root process sends its data to process 0, which applies the
        reduction operator (by default, summation). Then process 0 sends the
        reduced result back to all processes.
        
        The transfer cost is computed as:
          - For non-root processes: one send and one receive.
          - For the root process: (n-1) receives and (n-1) sends.
        """
        #TODO: Your code here
        self.comm.Reduce(src_array, dest_array, op, root=0)
        # broadcasting
        if self.Get_rank() == 0:
            # Process 0 sends the data to all other processes
            for i in range(1, self.Get_size()):
                self.comm.Send(dest_array, dest=i, tag=99)
        else:
            # All other processes receive the data
            self.comm.Recv(dest_array, source=0, tag=99)



    def myAlltoall(self, src_array, dest_array):
        """
        A manual implementation of all-to-all where each process sends a
        distinct segment of its source array to every other process.
        
        It is assumed that the total length of src_array (and dest_array)
        is evenly divisible by the number of processes.
        
        The algorithm loops over the ranks:
          - For the local segment (when destination == self), a direct copy is done.
          - For all other segments, the process exchanges the corresponding
            portion of its src_array with the other process via Sendrecv.
            
        The total data transferred is updated for each pairwise exchange.
        """
        #TODO: Your code here

        # # Sending source array to other processors
        # for i in range(self.Get_size()):
        #     if i != self.Get_rank():
        #         self.comm.Send(src_array, dest=i, tag=99)

        # tmp = np.empty_like(dest_array)

        # for i in range(self.Get_size()):
        #     if i == self.Get_rank():
        #         dest_array[i] = src_array[i]
        #     else:
        #         self.comm.Recv(tmp, source=i, tag=99)
        #         dest_array[i] = tmp[i]


        nprocs = self.comm.Get_size()
        rank = self.comm.Get_rank()

        # Determine the size of each segment to be sent
        seg_size = len(src_array) // nprocs


        recv_chunk = np.empty(seg_size, dtype=src_array.dtype)  # Buffer for received data

        for i in range(nprocs):
            send_chunk = src_array[i * seg_size: (i + 1) * seg_size]  # Extract segment to send

            if i == rank:
                # Copy local data directly (no communication needed)
                dest_array[i * seg_size: (i + 1) * seg_size] = send_chunk
            else:
                
                # Perform Sendrecv to exchange data
                self.comm.Sendrecv(send_chunk, dest=i, sendtag=99, 
                                recvbuf=recv_chunk, source=i, recvtag=99)
                
                # Store received data
                dest_array[i * seg_size: (i + 1) * seg_size] = recv_chunk