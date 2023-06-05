---
layout: post
title:  "SPMM on GPU"
date:   2023-5-30
tags: SPMM
subtitle: "SPMM on GPU"
description: 'SPMM on GPU'
color: 'rgb(154,133,255)'
cover: '../images/spmm.png'

---

**SPMM(稀疏矩阵乘)**

问题定义：

给定一稀疏矩阵 $A$ 和稠密矩阵 $B$, 求 $C = A * B$
假设 A.size = [row_num, row_num], nnz为A中非零元素的数目

B.size = [row_num, col_num], 当col_num为1的时候，SPMM即为SPMV(稀疏矩阵-向量乘)

稀疏矩阵的存储格式一般采用CSR或者CSC的格式，即为:

```c++
struct spmat{
  uint64_t rowptr[row_num+1]; 
  uint64_t col[nnz];
  float value[nnz];
};
```

在这种存储格式下, 稀疏矩阵第i行的所有非零元素为 $ col[rowptr[i] : rowptr[i+1]]$

B是一个稠密矩阵。SPMM的串行版本如下:

```c++
for(int ii = 0; ii < row_num; ++ii){
  uint64_t start = row_ptr[ii];
  uint64_t end = row_ptr[ii+1];
  float result[col_num];
  for(int kk = 0; kk < col_num; ++kk){
    for(int jj = start; jj < end; ++jj){
    	result[kk] += B[col[jj]][kk] * value[jj];
  	}  
    C[ii][kk] = result[kk];
   }
}
```

根据稀疏矩阵的划分方法不同，有两种主流的划分work balance的方法，即基于row和基于nnz的。

#### 基于row-balance的的SPMM

最容易想到的划分方法是基于row的划分方法，使用一个并行单位处理一行。注意由于col_num可能会很大，在col_num上也需要进行划分。

比如使用一个线程处理一行中的一列的算法, 一共需要row_num * col_num个线程

```c++
__global__ void spmm_row_balance_thread_kernel(const spmat *A, const float *B, const float *C)
{
    uint64_t row = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t col = blockIdx.y;

    float result = 0;

    uint64_t start = A.rowptr[row];
    uint64_t end = A.rowptr[row+1];

    for(int ii = start; ii < end; ++ii){
        uint64_t colIdx = A.col[ii];
        float val = A.value[ii];
        result += val * B[colidx*col_num+col];
    }
    C[row*col_num+col] = result;
}
const size_t THREADS = 256;
auto BLOCKS = dim3(row_num / THREADS, col_num);
spmm_row_balance_thread_kernel<<<BLOCKS, THREADS>>>(A, B, C);
```

使用warp的方法

1. 考虑到稀疏矩阵的每一行的非零元素在计算稠密矩阵的列时都要反复利用，可以将每行非零元素使用warp读取，然后shfl到整个warp中。实现更高的内存访问效率。这种情况warp中的每一个线程计算了稠密矩阵的每一列

```c++
__global__ void spmm_row_balance_warp_kernel(const spmat *A, const float *B, const float *C)
{
  int thread_id = blockIdx.x + blockDim.x + threadIdx.x;
  
  int row = thread_id >> 5;
  int lane_id = thread_id & (32-1);
  int col_idx = blockIdx.y << 5 + lane_id;
  int left_over = col_num - blockIdx.y << 5;
  
  int start = A.rowptr[row];
  int end = A.rowptr[row+1];
  
  int mat_row, mat_rows[32];
  float val, vals[32];
  int eid = start + lane_id;
  float result;
  
  for(int ii = start; ii < end; ii += 32){
		if(ii < end){
      mat_row = A.col[ii] * col_num;
      val = A.value[ii];
    }else{
      mat_row = -1;
      val = 0;
    }
    eid += 32;
    #pragma unroll
    for(int jj = 0; jj < 32; ++jj){
      mat_rows[jj] = __shfl_sync(FULL_MASK, mat_row, jj);
      vals[jj] = __shfl_sync(FULL_MASK, val, jj);
    }
    #pragma unroll
    for(int jj = 0; jj < 32; ++jj){
      if(lane_id < leftover && mat_rows[jj] != -1){
        val = vals[jj] * B[mat_rows[jj]+col_idx];
        result += val;
      }
    }
  }
  
  if(lane_id < leftover){
    C[row*col_num+col_idx] = result;
  }
}
const size_t THREADS = 256;
auto BLOCKS = dim3(row_num*WARP_SIZE / THREADS, col_num / 32);
spmm_row_balance_thread_kernel<<<BLOCKS, THREADS>>>(A, B, C);
```

2. 也可以利用warp来做reduction，这样做reduction会将结果最后放到warp的第一个线程中，结果是只有第一个线程做最后的写入

```c++
__global__ void spmm_row_balance_warp_reduction_kernel(const spmat *A, const float *B, const float *C)
{
  int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  int row = thread_id >> 5;
  int lane_id = thread_id & (32-1);
  int colIdx = blockIdx.y << 5;
  
  int start = A.rowptr[row]
  int end = A.rowptr[row+1];

  int k;
  float v;
  float result[32];
  
  for(int jj = start+lane_id; jj < end; jj += 32)
  {
    k = A.col[jj];
    v = A.value[jj];
    #pragma unroll
    for(int kk = 0; kk < 32; kk++){
        if(kk + colIdx < col_num)
            result[kk] = v * B[k*col_num+colIdx+kk];
    }
  }
  #pragma unroll
  for(int kk = 0; kk < 32; kk++){
    if(kk + colIdx < col_num)
        SHFL_DOWN_REDUCE(result[kk]);
  }
  if(lane_id == 0){
    for(int kk = 0; k < 32; kk++)
        C[row*col_num+colIdx+kk] = result[kk];
  }
}
const size_t THREADS = 256;
auto BLOCKS = dim3(row_num*WARP_SIZE / THREADS, col_num / 32);
spmm_row_blance_warp_kernel<<<BLOCKS, THREADS>>>(A, B, C);
```



很明显，这种划分方法会导致在那些skewed-distribution的稀疏数据上workload非常不平衡，即一个线程或一个warp要处理的start和end非常长。可以使用基于nnz的划分方法来解决这个问题。

#### 基于nnz-balance的SPMM

这种方法使用一个线程处理固定数目的nnz(非零元素)，比如使用一个线程处理32个nnz。
很显然，一个线程中的若干个nnz不是都属于稀疏矩阵的同一个row, 因此需要付出额外的代价来寻找nnz所属的行。
另外，最后对C矩阵的更改也需要atomic来保证不同的线程更改C矩阵的正确性。

以下是使用一个线程处理4个nnz的示例, 即一共需要(nnz/32) * col_num个线程

```c++
const int NE_PER_THREAD = 32;
__global__ void spmm_nnz_balance_thread_kernel(const spmat *A, const float *B, const float *C)
{
    int64_t nnz_start = blockIdx.x * blockDim.x + threadIdx.x;
    if(nnz_start > (nnz + NE_PER_THREAD - 1) / NE_PER_THREAD) return;
    int64_t col = blockIdx.y;

    float result = 0;

    int eid = nnz_start;
    int row = binary_search_segment_number(A.rowptr, A.col, A.nnz, eid);
    int step = A.rowptr[row+1] - eid;

    uint64_t colIdx = 0;
    float value = 0.0;

    for(int ii = 0; ii < NE_PER_THREAD; ++ii){
        if(eid > A.nnz) break;
        if(ii < step){
            colIdx = A.col[eid];
            value = A.value[colIdx];
            result += value * B[colIdx*col_num+col];
            eid++;
        }else{
            atomicAdd(&C[row*col_num+col], result);
            row = binary_search_segment_number(A.rowptr, A.col, A.nnz, eid);
            step = A.rowptr[row+1]-eid+ii;
            colIdx = A.col[eid];
            value = A.value[colIdx];
            result = value * B[colIdx*col_num+col];
            eid++;
        }
    }
    atomicAdd(&C[row*col_num+col], result);
}
const size_t THREADS = 256;
auto BLOCKS = dim3(nnz/(NE_PER_THREAD*THREADS), col_num);
spmm_nnz_balance_thread_kernel<<<BLOCKS, THREADS>>>(A, B, C);
```

显然，这种方法会在稀疏矩阵计算每一列的时候都去做行的search, 这部分开销是很大的，我们可以使用一个线程处理稠密矩阵中的若干列，这样只需要计算一次search。如处理32列:

```c++
const int NE_PER_THREAD = 32;
__global__ void spmm_nnz_balance_coltile_kernel(const spmat *A, const float *B, const float *C)
{
    int64_t nnz_start = blockIdx.x * blockDim.x + threadIdx.x;
    if(nnz_start > (nnz + NE_PER_THREAD - 1) / NE_PER_THREAD) return;
    int64_t col = blockIdx.y * 32;
    

    float result[32] = {0.0};

    int eid = nnz_start;
    int row = binary_search_segment_number(A.rowptr, A.col, A.nnz, eid);
    int step = A.rowptr[row+1] - eid;

    uint64_t colIdx = 0;
    float value = 0.0;

    for(int ii = 0; ii < NE_PER_THREAD; ++ii){
        if(eid > A.nnz) break;
        if(ii < step){
            colIdx = A.col[eid];
            value = A.value[colIdx];
            #pragma unroll
            for(int jj = 0; jj < 32; ++jj){
                if(col+jj < col_num)
                    result[jj] += value * B[colIdx*col_num+col+jj];
            }
            eid++;
        }else{
            #pragma unroll
            for(int jj = 0; jj < 32; ++jj){
                if(col+jj < col_num)
                    atomicAdd(&C[row*col_num+col+jj], result);
            }
            row = binary_search_segment_number(A.rowptr, A.col, A.nnz, eid);
            step = A.rowptr[row+1]-eid+ii;
            colIdx = A.col[eid];
            value = A.value[colIdx];
            #pragma unroll
            for(int jj = 0; jj < 32; ++jj){
                if(col+jj < col_num)
                    result = value * B[colIdx*col_num+col+jj];
            }
            eid++;
        }
    }
    #pragma unroll
    for(int jj = 0; jj < 32; ++jj){
        if(col+jj < col_num)
            atomicAdd(&C[row*col_num+col+jj], result);
    }
}
const size_t THREADS = 256;
auto BLOCKS = dim3(nnz/(NE_PER_THREAD/THREADS), (col_num+31) / 32);
spmm_nnz_balance_coltile_kernel<<<BLOCKS, THREADS>>>(A, B, C);
```

考虑使用一个warp处理若干个非零元素，warp内部可以利用warp之间的通信。

```c++
const int NE_PER_WARP = 32;
const size_t THREADS = 256;
const size_t WARP_SIZE = 32;
const size_t COL_TILE = 32;
__global__ void spmm_nnz_balance_warp_kernel(const spmat *A, const float *B, const float *C){
    uint64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * 32;
    
    int warp_idx = thread_id >> 5; // thread_id / 32 nnz_start
    if(warp_idx > (nnz + NE_PER_WARP - 1) / NE_PER_WARP) return;
    int lane_idx = thread_id & (32-1); // thread % 32
    
    float k;
    float v;
    float buffer[COL_TILE] = {0.0};
    float result[COL_TILE] = {0.0};


    for(int eid = warp_idx+lane_idx; eid < warp_idx + NE_PER_WARP; eid += NE_PER_WARP / WARP_SIZE){
        int row = binary_search_segment_number(A.rowptr, A.col, A.nnz, eid);
        if(eid < A.nnz){
            k = A.col[eid];
            v = A.value[eid];
        }else{
            k = 0;
            v = 0.0f;
        }
        buffer = (B + row * col_num + col);
        #pragma unroll
        for(int jj = 0; jj < COL_TILE; ++jj)
        {
            result[jj] = buffer[jj] * v;
        }
        int warp_start = __shfl_sync(FULL_MASK, row, 31);
        int warp_end = __shfl_sync(FULL_MASK, row, 0);
        if(warp_end == warp_start){
            //if all the element in a warp belong to the same row 
            //parallel reduction
            #pragma unroll
            for(int jj = 0; jj < COL_TILE; jj++){
                SHFL_DOWN_REDUCE(result[jj]);
            }
            if(lane_id == 0){
                #pragma unroll
                for(int jj = 0; jj < COL_TILE; ++jj){
                    if(jj+col < col_num)
                        atomicAdd(&C[row*col_num+col+jj], result[jj]);
                }
            }
        }else{
            bool is_seg_start = (__shfl_up_sync(FULL_MASK, row, 1) != row) || (lane_id == 0);
            float tmpv;
            int tmpr;
            #pragma unroll
            for(int jj = 0; jj < COL_TILE; ++jj){
                SEG_SHFL_SCAN(result[i], tmpv, row, tmpr);
            }
            if(is_seg_start){
                #pragma unroll
                for(int jj = 0; jj < COL_TILE; ++jj){
                    if(jj+col < col_num)
                        atomicAdd(&C[row*col_num+col+jj], result[i]);
                }
            }
        }
    }
    return;
}

const size_t nnzDim = (nnz + NE_PER_WARP - 1) / NE_PER_WARP;
const size_t colDim = (col_num + 31) / 32;
auto BLOCK = dim3((nnzDim*WARP_SIZE) / THREADS, colDim);
spmm_nnz_balance_warp_kernel<<<BLOCKS, THREADS>>>(A, B, C);
```

nnz的row-cache方法

```c++
const int NE_PER_WARP = 32;
const size_t THREADS = 256;
const size_t WARP_SIZE = 32;
const size_t COL_TILE = 32;
__global__ void spmm_nnz_balance_rowcache_kernel(const spmat *A, const float *B, const float *C){
    uint64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * 32;
    
    int warp_idx = thread_id >> 5; // thread_id / 32 nnz_start
    if(warp_idx > (nnz + NE_PER_WARP - 1) / NE_PER_WARP) return;
    int lane_idx = thread_id & (32-1); // thread % 32
    int mat_col_idx = lane_idx + col;
    
    int mat_row, mat_rows[32];
    int row_id, row_ids[32];
    float v, vals[32];

    float result[32] = {0.0};

    for(int eid = warp_idx+lane_idx; eid < warp_idx + NE_PER_WARP; eid += NE_PER_WARP / WARP_SIZE){
        if(eid < A.nnz){
            mat_row = A.col[eid]*col_num;
            val = A.value[eid];
        }else{
            mat_row = 0;
            val = 0.0f;
        }
        row_id = binary_search_segment_number(A.rowptr, A.col, A.nnz, eid);
        #pragma unroll
        for(int jj = 0; jj < 32; jj++){
            mat_rows[jj] = __shfl_sync(FULL_MASK, mat_row, jj);
            row_ids[jj] = __shfl_sync(FULL_MASK, row_id, jj);
            vals[jj] = __shfl_sync(FULL_MASK, val, jj);
        }
        #pragma unroll
        for(int jj = 0; jj < 32; jj++){
            if(mat_rows[jj] != -1){
                val = __ldg(B+mat_rows[jj]+mat_col_idx);
                val = val * vals[jj];
                result[jj] = val;
            }
        }
        //scan
        int row_curr = row_ids[0], next_row;
        float result = result[0];
        #pragma unroll
        for(int jj = 1; jj < 32; jj++){
            next_row = row_ids[jj];
            if(row_curr != next_row){
                atomicAdd(&C[row_curr*col_num+mat_col_idx], result);
                row_curr = next_row;
                result = result[jj];
            }
            result = result + results[jj];
        }
        atomicAdd(&C[row_curr*col_num+mat_col_idx], result);
    }
    return;
}

const size_t nnzDim = (nnz + NE_PER_WARP - 1) / NE_PER_WARP;
const size_t colDim = (col_num + 31) / 32;
auto BLOCK = dim3((nnzDim*WARP_SIZE) / THREADS, colDim);
spmm_nnz_balance_warp_kernel<<<BLOCKS, THREADS>>>(A, B, C);
```

#### Summary
在SPMM这个算法中有3个需要考虑的因素，

1. workload balance的划分方法: 按行划分或按nnz划分
    选择哪一种的因素取决的稀疏矩阵中每一行的degree的分布，如果每行的非零元素都

2. 做reduction的方法，
    a) sequential reduction
    b) warp parallel reduction
    c) row_cache将每一行的非零元素shuffle到每个线程中
    选择哪一种取决于稠密矩阵的列的数量，选择的条件是
    N > 32: row_cache
    32 > N > 4: sequential
    4 > N: parallel reduction
    







