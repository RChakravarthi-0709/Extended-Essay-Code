#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <cuda.h>

#define AES_BLOCK_SIZE 16
#define AES_ROUNDS 10

// AES S-box
__device__ __constant__ unsigned char sbox[256] = {
    // full AES S-box values here...
};

// Round constants
__device__ __constant__ unsigned char rcon[10] = {
    0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80,0x1B,0x36
};

// --- AES helper functions ---
__device__ unsigned char gmul(unsigned char a, unsigned char b) {
    unsigned char p = 0;
    for (int i = 0; i < 8; i++) {
        if (b & 1) p ^= a;
        bool hi_bit_set = (a & 0x80);
        a <<= 1;
        if (hi_bit_set) a ^= 0x1b;
        b >>= 1;
    }
    return p;
}

__device__ void SubBytes(unsigned char* state) {
    for (int i = 0; i < AES_BLOCK_SIZE; i++)
        state[i] = sbox[state[i]];
}

__device__ void ShiftRows(unsigned char* state) {
    unsigned char tmp[AES_BLOCK_SIZE];
    tmp[0]=state[0]; tmp[1]=state[5]; tmp[2]=state[10]; tmp[3]=state[15];
    tmp[4]=state[4]; tmp[5]=state[9]; tmp[6]=state[14]; tmp[7]=state[3];
    tmp[8]=state[8]; tmp[9]=state[13]; tmp[10]=state[2]; tmp[11]=state[7];
    tmp[12]=state[12]; tmp[13]=state[1]; tmp[14]=state[6]; tmp[15]=state[11];
    for (int i=0;i<AES_BLOCK_SIZE;i++) state[i]=tmp[i];
}

__device__ void MixColumns(unsigned char* state) {
    for (int i=0;i<4;i++) {
        int col=i*4;
        unsigned char a0=state[col], a1=state[col+1], a2=state[col+2], a3=state[col+3];
        state[col]   = gmul(a0,2)^gmul(a1,3)^a2^a3;
        state[col+1] = a0^gmul(a1,2)^gmul(a2,3)^a3;
        state[col+2] = a0^a1^gmul(a2,2)^gmul(a3,3);
        state[col+3] = gmul(a0,3)^a1^a2^gmul(a3,2);
    }
}

__device__ void AddRoundKey(unsigned char* state, const unsigned char* roundKey) {
    for (int i=0;i<AES_BLOCK_SIZE;i++) state[i]^=roundKey[i];
}

// AES block encrypt
__device__ void aes_encrypt_block(unsigned char* block, const unsigned char* roundKeys) {
    AddRoundKey(block, roundKeys);
    for (int round=1; round<AES_ROUNDS; round++) {
        SubBytes(block);
        ShiftRows(block);
        MixColumns(block);
        AddRoundKey(block, roundKeys+round*AES_BLOCK_SIZE);
    }
    SubBytes(block);
    ShiftRows(block);
    AddRoundKey(block, roundKeys+AES_ROUNDS*AES_BLOCK_SIZE);
}

// --- GHASH (simplified) ---
__device__ void ghash_mul(unsigned char* X, const unsigned char* H) {
    // Simplified GF(2^128) multiplication
    for (int i=0;i<AES_BLOCK_SIZE;i++) {
        X[i]^=H[i]; // placeholder for full carry-less multiply
    }
}

// --- AES-GCM kernel ---
__global__ void aes_gcm_encrypt_kernel(unsigned char* plaintext,
                                       unsigned char* ciphertext,
                                       const unsigned char* roundKeys,
                                       const unsigned char* iv,
                                       const unsigned char* H,
                                       size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size / AES_BLOCK_SIZE) {
        unsigned char block[AES_BLOCK_SIZE];
        unsigned char counter[AES_BLOCK_SIZE];

        // Load plaintext
        for (int i=0;i<AES_BLOCK_SIZE;i++)
            block[i]=plaintext[idx*AES_BLOCK_SIZE+i];

        // Derive counter from IV + idx
        for (int i=0;i<AES_BLOCK_SIZE;i++) counter[i]=iv[i%12];
        counter[12]=(idx>>24)&0xFF;
        counter[13]=(idx>>16)&0xFF;
        counter[14]=(idx>>8)&0xFF;
        counter[15]=idx&0xFF;

        // Encrypt counter
        aes_encrypt_block(counter, roundKeys);

        // XOR with plaintext
        for (int i=0;i<AES_BLOCK_SIZE;i++)
            ciphertext[idx*AES_BLOCK_SIZE+i]=block[i]^counter[i];

        // GHASH update
        ghash_mul(ciphertext+idx*AES_BLOCK_SIZE,H);
    }
}

// --- Host benchmark ---
int main() {
    std::vector<size_t> payloadSizes = {
        16*1024ULL, 512*1024ULL, 8ULL*1024*1024,
        64ULL*1024*1024, 512ULL*1024*1024, 1ULL*1024*1024*1024
    };

    std::mt19937 rng(std::random_device{}());
    std::ofstream csv("gpu_results.csv");
    csv<<"PayloadSize(Bytes),Trial,Latency(us),Throughput(Bps),Mode\n";

    for (size_t payloadSize:payloadSizes) {
        std::vector<double> latKernel, thrKernel, latEnd, thrEnd;

        for (int trial=1;trial<=50;trial++) {
            // Host buffers
            std::vector<unsigned char> h_plain(payloadSize);
            std::vector<unsigned char> h_cipher(payloadSize);
            std::vector<unsigned char> h_key(16), h_iv(12), h_H(16);

            for (auto &b:h_plain) b=rng()&0xFF;
            for (auto &b:h_key) b=rng()&0xFF;
            for (auto &b:h_iv) b=rng()&0xFF;
            for (auto &b:h_H) b=rng()&0xFF;

            // Device buffers
            unsigned char *d_plain,*d_cipher,*d_roundKeys,*d_iv,*d_H;
            cudaMalloc(&d_plain,payloadSize);
            cudaMalloc(&d_cipher,payloadSize);
            cudaMalloc(&d_roundKeys,(AES_ROUNDS+1)*AES_BLOCK_SIZE);
            cudaMalloc(&d_iv,h_iv.size());
            cudaMalloc(&d_H,h_H.size());

            cudaMemcpy(d_plain,h_plain.data(),payloadSize,cudaMemcpyHostToDevice);
            cudaMemcpy(d_iv,h_iv.data(),h_iv.size(),cudaMemcpyHostToDevice);
            cudaMemcpy(d_H,h_H.data(),h_H.size(),cudaMemcpyHostToDevice);

            // Expand key on host (simplified: just copy key as roundKeys)
            std::vector<unsigned char> roundKeys((AES_ROUNDS+1)*AES_BLOCK_SIZE);
            for (int i=0;i<16;i++) roundKeys[i]=h_key[i];
            cudaMemcpy(d_roundKeys,roundKeys.data(),roundKeys.size(),cudaMemcpyHostToDevice);

            int threads=256;
            int blocks=(payloadSize/AES_BLOCK_SIZE+threads-1)/threads;

            // Kernel-only timing
            cudaEvent_t start,stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);

            aes_gcm_encrypt_kernel<<<blocks,threads>>>(d_plain,d_cipher,d_roundKeys,d_iv,d_H,payloadSize);

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float ms;
            cudaEventElapsedTime(&ms,start,stop);
            double lat_us=ms*1000.0;
            double thr=static_cast<double>(payloadSize)/(lat_us/1e6);
            latKernel.push_back(lat_us);
            thrKernel.push_back(thr);
            csv<<payloadSize<<","<<trial<<","<<lat_us<<","<<thr<<",GPU-Kernel\n";

            // End-to-end timing
            cudaEventRecord(start);
            cudaMemcpy(d_plain,h_plain.data(),payloadSize,cudaMemcpyHostToDevice);
            aes_gcm_encrypt_kernel<<<blocks,threads>>>(d_plain,d_cipher,d_roundKeys,d_iv,d_H,payloadSize);
            cudaMemcpy(h_cipher.data(),d_cipher,payloadSize,cudaMemcpyDeviceToHost);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&ms,start,stop);
            lat_us=ms*1000.0;
            thr=static_cast<double>(payloadSize)/(lat_us/1e6);
