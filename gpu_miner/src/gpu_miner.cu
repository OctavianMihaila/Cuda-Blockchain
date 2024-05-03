#include <stdio.h>
#include <stdint.h>
#include "../include/utils.cuh"
#include <string.h>
#include <stdlib.h>
#include <inttypes.h>

// TODO: Implement function to search for all nonces from 1 through MAX_NONCE (inclusive) using CUDA Threads
__global__ void findNonce(uint32_t *nonce, BYTE *block_content, size_t current_length, BYTE *block_hash, BYTE *difficulty) {
    uint32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t step = gridDim.x * blockDim.x;

    for (uint32_t nonce_to_try = thread_id; nonce_to_try <= MAX_NONCE; nonce_to_try += step) {
        char nonce_as_string[SHA256_HASH_SIZE];
        intToString(nonce_to_try, nonce_as_string);
	
	BYTE new_block_content[BLOCK_SIZE];
        // Copy block_content to new_block_content
        memcpy(new_block_content, block_content, current_length);
        
        // Add nonce to block_content
        memcpy(new_block_content + current_length, nonce_as_string, d_strlen(nonce_as_string) + 1);
   
        BYTE new_block_hash[SHA256_HASH_SIZE];

        apply_sha256(new_block_content, current_length + d_strlen(nonce_as_string), new_block_hash, 1);
        
        // compare block_hash with difficulty_5_zeros
        if (compare_hashes(new_block_hash, difficulty) <= 0) {
            if (nonce_to_try < *nonce || *nonce == 0) {
                atomicExch(nonce, nonce_to_try);
                memcpy(block_hash, new_block_hash, SHA256_HASH_SIZE);
            }
            return;
        }
    }
}

int main(int argc, char **argv) {
    BYTE hashed_tx1[SHA256_HASH_SIZE], hashed_tx2[SHA256_HASH_SIZE], hashed_tx3[SHA256_HASH_SIZE], hashed_tx4[SHA256_HASH_SIZE],
            tx12[SHA256_HASH_SIZE * 2], tx34[SHA256_HASH_SIZE * 2], hashed_tx12[SHA256_HASH_SIZE], hashed_tx34[SHA256_HASH_SIZE],
            tx1234[SHA256_HASH_SIZE * 2], top_hash[SHA256_HASH_SIZE], block_content[BLOCK_SIZE];
    BYTE block_hash[SHA256_HASH_SIZE];// = "0000000000000000000000000000000000000000000000000000000000000000"; // TODO: Update
    uint32_t nonce; // TODO: Update
    size_t current_length;

    memset(block_hash, 0, SHA256_HASH_SIZE);

    // Top hash
    apply_sha256(tx1, strlen((const char*)tx1), hashed_tx1, 1);
    apply_sha256(tx2, strlen((const char*)tx2), hashed_tx2, 1);
    apply_sha256(tx3, strlen((const char*)tx3), hashed_tx3, 1);
    apply_sha256(tx4, strlen((const char*)tx4), hashed_tx4, 1);
    strcpy((char *)tx12, (const char *)hashed_tx1);
    strcat((char *)tx12, (const char *)hashed_tx2);
    apply_sha256(tx12, strlen((const char*)tx12), hashed_tx12, 1);
    strcpy((char *)tx34, (const char *)hashed_tx3);
    strcat((char *)tx34, (const char *)hashed_tx4);
    apply_sha256(tx34, strlen((const char*)tx34), hashed_tx34, 1);
    strcpy((char *)tx1234, (const char *)hashed_tx12);
    strcat((char *)tx1234, (const char *)hashed_tx34);
    apply_sha256(tx1234, strlen((const char*)tx34), top_hash, 1);

    // prev_block_hash + top_hash
    strcpy((char*)block_content, (const char*)prev_block_hash);
    strcat((char*)block_content, (const char*)top_hash);
    current_length = strlen((char*) block_content);

    BYTE *d_block_content, *d_block_hash;
    uint32_t *d_nonce;
    BYTE *d_difficulty;
    
    cudaMalloc((void**)&d_block_content, BLOCK_SIZE);
    cudaMalloc((void**)&d_block_hash, SHA256_HASH_SIZE);
    cudaMalloc((void**)&d_nonce, sizeof(uint32_t));
    cudaMalloc((void**)&d_difficulty, SHA256_HASH_SIZE);

    cudaMemcpy(d_block_content, block_content, BLOCK_SIZE, cudaMemcpyHostToDevice);

    cudaMemcpy(d_difficulty, difficulty_5_zeros, SHA256_HASH_SIZE, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    startTiming(&start, &stop);

    findNonce<<<112, 512>>>(d_nonce, d_block_content, current_length, d_block_hash, d_difficulty);

    cudaDeviceSynchronize();

    cudaMemcpy(&nonce, d_nonce, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(block_hash, d_block_hash, SHA256_HASH_SIZE, cudaMemcpyDeviceToHost);

    float seconds = stopTiming(&start, &stop);
    printResult(block_hash, nonce, seconds);

    cudaFree(d_block_content);
    cudaFree(d_block_hash);
    cudaFree(d_nonce);
    
    return 0;
}
