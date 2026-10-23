#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <cuda.h>

#define AES_BLOCK_SIZE 16
#define AES_ROUNDS 10

// AES S-Box
__device__ __constant__ unsigned char sbox[256] = {
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,
    0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,
    0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,
    0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,
    0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,
    0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,
    0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,
    0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,
    0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,
    0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,
    0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,
    0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,
    0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,
    0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,
    0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,
    0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,
    0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16
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
