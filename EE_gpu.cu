// GPU-only benchmark for an IB Computer Science EE.
//
// The GPU performs AES-128 key expansion, counter-mode encryption AND
// GHASH/tag generation. OpenSSL is used only as an untimed correctness reference;
// CPU latency and throughput are not benchmarked or written to the results.
// This is intended as a transparent educational implementation, not production
// cryptographic software. It uses a straightforward bit-serial GF(2^128)
// multiplier; it is correct in structure but may be slow compared with highly
// optimized implementations such as OpenSSL.
//
// Compile from the x64 Native Tools Command Prompt for Visual Studio on Windows:
//   nvcc -O3 -std=c++17 -arch=sm_86 -I"%USERPROFILE%\vcpkg\installed\x64-windows\include" "EE_gpu_aes_gcm_gpu_only.cu" -o "EE_gpu_only.exe" "%USERPROFILE%\vcpkg\installed\x64-windows\lib\libcrypto.lib"
// Smoke test 16 KiB and 8 MiB payloads, one trial each:
//   EE_gpu_only.exe --smoke
// Full run: six payload sizes, 50 trials each:
//   EE_gpu_only.exe
//
// Payload sizes in this file are all multiples of 16*256 bytes. This lets the
// chunked GHASH reduction use equal-sized chunks at every non-final reduction
// stage. Do not add arbitrary payload sizes without extending that handling.

#include <cuda_runtime.h>
#include <openssl/evp.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <climits>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#define AES_BLOCK_SIZE 16
#define AES_ROUNDS 10
#define CHUNK_BLOCKS 256
#define REDUCE_FANIN 256

#define CUDA_CHECK(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error at ") + __FILE__ + ":" + \
            std::to_string(__LINE__) + ": " + cudaGetErrorString(err__)); \
    } \
} while (0)

__device__ __constant__ unsigned char SBOX_DEVICE[256] = {
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16
};

__device__ __forceinline__ unsigned char aes_gmul(unsigned char a, unsigned char b) {
    unsigned char p = 0;
    for (int i = 0; i < 8; ++i) {
        if (b & 1) p ^= a;
        const bool high = (a & 0x80) != 0;
        a = static_cast<unsigned char>(a << 1);
        if (high) a ^= 0x1b;
        b >>= 1;
    }
    return p;
}

__device__ void sub_bytes(unsigned char state[16]) {
    for (int i = 0; i < 16; ++i) state[i] = SBOX_DEVICE[state[i]];
}

__device__ void shift_rows(unsigned char state[16]) {
    unsigned char tmp[16];
    tmp[0]=state[0];   tmp[1]=state[5];   tmp[2]=state[10];  tmp[3]=state[15];
    tmp[4]=state[4];   tmp[5]=state[9];   tmp[6]=state[14];  tmp[7]=state[3];
    tmp[8]=state[8];   tmp[9]=state[13];  tmp[10]=state[2]; tmp[11]=state[7];
    tmp[12]=state[12]; tmp[13]=state[1]; tmp[14]=state[6]; tmp[15]=state[11];
    for (int i = 0; i < 16; ++i) state[i] = tmp[i];
}

__device__ void mix_columns(unsigned char state[16]) {
    for (int i = 0; i < 4; ++i) {
        const int col = i * 4;
        const unsigned char a0 = state[col];
        const unsigned char a1 = state[col + 1];
        const unsigned char a2 = state[col + 2];
        const unsigned char a3 = state[col + 3];
        state[col]     = aes_gmul(a0,2) ^ aes_gmul(a1,3) ^ a2 ^ a3;
        state[col + 1] = a0 ^ aes_gmul(a1,2) ^ aes_gmul(a2,3) ^ a3;
        state[col + 2] = a0 ^ a1 ^ aes_gmul(a2,2) ^ aes_gmul(a3,3);
        state[col + 3] = aes_gmul(a0,3) ^ a1 ^ a2 ^ aes_gmul(a3,2);
    }
}

__device__ void add_round_key(unsigned char state[16], const unsigned char* roundKey) {
    for (int i = 0; i < 16; ++i) state[i] ^= roundKey[i];
}

__device__ void aes_encrypt_block(unsigned char block[16], const unsigned char* roundKeys) {
    add_round_key(block, roundKeys);
    for (int round = 1; round < AES_ROUNDS; ++round) {
        sub_bytes(block);
        shift_rows(block);
        mix_columns(block);
        add_round_key(block, roundKeys + round * 16);
    }
    sub_bytes(block);
    shift_rows(block);
    add_round_key(block, roundKeys + AES_ROUNDS * 16);
}

__device__ __constant__ unsigned char RCON_DEVICE[10] = {
    0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80,0x1B,0x36
};

// Key expansion is run on the GPU so the benchmark's key schedule is included
// in the GPU kernel pipeline rather than being silently prepared on the CPU.
__global__ void aes128_key_expansion_kernel(const unsigned char* key,
                                           unsigned char* roundKeys) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    for (int i = 0; i < 16; ++i) roundKeys[i] = key[i];

    int generated = 16;
    int rconIndex = 0;
    unsigned char temp[4];
    while (generated < 176) {
        for (int i = 0; i < 4; ++i) temp[i] = roundKeys[generated - 4 + i];
        if (generated % 16 == 0) {
            const unsigned char first = temp[0];
            temp[0] = static_cast<unsigned char>(SBOX_DEVICE[temp[1]] ^ RCON_DEVICE[rconIndex++]);
            temp[1] = SBOX_DEVICE[temp[2]];
            temp[2] = SBOX_DEVICE[temp[3]];
            temp[3] = SBOX_DEVICE[first];
        }
        for (int i = 0; i < 4; ++i) {
            roundKeys[generated] = static_cast<unsigned char>(roundKeys[generated - 16] ^ temp[i]);
            ++generated;
        }
    }
}

// Big-endian 64-bit helpers used by GHASH. These avoid shifting a 16-byte
// array in every multiplication round, while preserving GCM's bit convention.
__device__ __forceinline__ uint64_t load_be64_device(const unsigned char* p) {
    return (static_cast<uint64_t>(p[0]) << 56) |
           (static_cast<uint64_t>(p[1]) << 48) |
           (static_cast<uint64_t>(p[2]) << 40) |
           (static_cast<uint64_t>(p[3]) << 32) |
           (static_cast<uint64_t>(p[4]) << 24) |
           (static_cast<uint64_t>(p[5]) << 16) |
           (static_cast<uint64_t>(p[6]) << 8)  |
            static_cast<uint64_t>(p[7]);
}

__device__ __forceinline__ void store_be64_device(unsigned char* p, uint64_t x) {
    p[0] = static_cast<unsigned char>(x >> 56);
    p[1] = static_cast<unsigned char>(x >> 48);
    p[2] = static_cast<unsigned char>(x >> 40);
    p[3] = static_cast<unsigned char>(x >> 32);
    p[4] = static_cast<unsigned char>(x >> 24);
    p[5] = static_cast<unsigned char>(x >> 16);
    p[6] = static_cast<unsigned char>(x >> 8);
    p[7] = static_cast<unsigned char>(x);
}

// GHASH multiplication in the bit convention specified for GCM.
// Inputs are 16-byte big-endian field elements. Output may not alias x or h.
__device__ void gf128_mul(const unsigned char* x, const unsigned char* h,
                          unsigned char out[16]) {
    const uint64_t xHi = load_be64_device(x);
    const uint64_t xLo = load_be64_device(x + 8);
    uint64_t vHi = load_be64_device(h);
    uint64_t vLo = load_be64_device(h + 8);
    uint64_t zHi = 0;
    uint64_t zLo = 0;

    for (int bit = 0; bit < 128; ++bit) {
        const bool set = bit < 64
            ? ((xHi >> (63 - bit)) & 1ULL) != 0
            : ((xLo >> (127 - bit)) & 1ULL) != 0;
        if (set) {
            zHi ^= vHi;
            zLo ^= vLo;
        }

        const bool lsb = (vLo & 1ULL) != 0;
        vLo = (vLo >> 1) | (vHi << 63);
        vHi >>= 1;
        if (lsb) vHi ^= 0xe100000000000000ULL;
    }

    store_be64_device(out, zHi);
    store_be64_device(out + 8, zLo);
}

// Derive H = AES_K(0^128) and the tag mask AES_K(J0), with J0 = IV || 0x00000001.
__global__ void derive_gcm_values_kernel(const unsigned char* roundKeys,
                                          const unsigned char* iv,
                                          unsigned char* H,
                                          unsigned char* tagMask) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    unsigned char zero[16] = {0};
    aes_encrypt_block(zero, roundKeys);
    for (int i = 0; i < 16; ++i) H[i] = zero[i];

    unsigned char j0[16] = {0};
    for (int i = 0; i < 12; ++i) j0[i] = iv[i];
    j0[12] = 0; j0[13] = 0; j0[14] = 0; j0[15] = 1;
    aes_encrypt_block(j0, roundKeys);
    for (int i = 0; i < 16; ++i) tagMask[i] = j0[i];
}

// Each CUDA thread encrypts one counter block. The selected benchmark sizes
// are block-aligned, so no partial final block is needed here.
__global__ void aes_ctr_kernel(const unsigned char* plaintext,
                               unsigned char* ciphertext,
                               const unsigned char* roundKeys,
                               const unsigned char* iv,
                               size_t numBlocks) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= numBlocks) return;

    unsigned char counter[16];
    for (int i = 0; i < 12; ++i) counter[i] = iv[i];
    const uint32_t ctr = static_cast<uint32_t>(idx) + 2U; // first payload counter is inc32(J0) = IV || 2
    counter[12] = static_cast<unsigned char>((ctr >> 24) & 0xff);
    counter[13] = static_cast<unsigned char>((ctr >> 16) & 0xff);
    counter[14] = static_cast<unsigned char>((ctr >> 8) & 0xff);
    counter[15] = static_cast<unsigned char>(ctr & 0xff);

    aes_encrypt_block(counter, roundKeys);
    const size_t offset = idx * 16;
    for (int i = 0; i < 16; ++i) ciphertext[offset + i] = plaintext[offset + i] ^ counter[i];
}

// One thread computes a local GHASH for 256 ciphertext blocks. The resulting
// chunk hashes are combined by a tree of reduction kernels below.
__global__ void ghash_chunks_kernel(const unsigned char* ciphertext,
                                    const unsigned char* H,
                                    unsigned char* chunkHashes,
                                    size_t numBlocks) {
    const size_t chunk = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t numChunks = numBlocks / CHUNK_BLOCKS;
    if (chunk >= numChunks) return;

    unsigned char y[16] = {0};
    unsigned char x[16];
    unsigned char product[16];
    const size_t firstBlock = chunk * CHUNK_BLOCKS;

    for (size_t b = 0; b < CHUNK_BLOCKS; ++b) {
        const size_t offset = (firstBlock + b) * 16;
        for (int j = 0; j < 16; ++j) x[j] = static_cast<unsigned char>(y[j] ^ ciphertext[offset + j]);
        gf128_mul(x, H, product);
        for (int j = 0; j < 16; ++j) y[j] = product[j];
    }
    for (int j = 0; j < 16; ++j) chunkHashes[chunk * 16 + j] = y[j];
}

// Compute H^(segmentBlocks). Since segmentBlocks is 256^level, repeated
// squaring (8*level times) obtains the required field power.
__global__ void compute_h_power_kernel(const unsigned char* H,
                                       unsigned char* power,
                                       int squarings) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    unsigned char x[16], product[16];
    for (int i = 0; i < 16; ++i) x[i] = H[i];
    for (int i = 0; i < squarings; ++i) {
        gf128_mul(x, x, product);
        for (int j = 0; j < 16; ++j) x[j] = product[j];
    }
    for (int i = 0; i < 16; ++i) power[i] = x[i];
}

// Combine equal-sized GHASH segments. If each input hash represents m blocks,
// concatenation follows Y_new = (Y_old * H^m) XOR segmentHash.
__global__ void reduce_ghash_kernel(const unsigned char* inputHashes,
                                    unsigned char* outputHashes,
                                    const unsigned char* hPower,
                                    size_t inputCount) {
    const size_t outIndex = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t outputCount = (inputCount + REDUCE_FANIN - 1) / REDUCE_FANIN;
    if (outIndex >= outputCount) return;

    unsigned char y[16] = {0};
    unsigned char product[16];
    const size_t begin = outIndex * REDUCE_FANIN;
    size_t end = begin + REDUCE_FANIN;
    if (end > inputCount) end = inputCount;

    for (size_t i = begin; i < end; ++i) {
        gf128_mul(y, hPower, product);
        for (int j = 0; j < 16; ++j) {
            y[j] = static_cast<unsigned char>(product[j] ^ inputHashes[i * 16 + j]);
        }
    }
    for (int j = 0; j < 16; ++j) outputHashes[outIndex * 16 + j] = y[j];
}

// Append GCM's length block (no associated data), then create tag = AES_K(J0) XOR GHASH.
__global__ void finish_gcm_tag_kernel(const unsigned char* ciphertextHash,
                                      const unsigned char* H,
                                      const unsigned char* tagMask,
                                      uint64_t ciphertextBytes,
                                      unsigned char* tag) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    unsigned char lengthBlock[16] = {0};
    const uint64_t ciphertextBits = ciphertextBytes * 8ULL;
    for (int i = 0; i < 8; ++i) {
        lengthBlock[15 - i] = static_cast<unsigned char>((ciphertextBits >> (8 * i)) & 0xff);
    }

    unsigned char x[16], y[16];
    for (int i = 0; i < 16; ++i) x[i] = static_cast<unsigned char>(ciphertextHash[i] ^ lengthBlock[i]);
    gf128_mul(x, H, y);
    for (int i = 0; i < 16; ++i) tag[i] = static_cast<unsigned char>(y[i] ^ tagMask[i]);
}

struct ReferenceResult {
    std::vector<unsigned char> ciphertext;
    std::array<unsigned char, 16> tag;
};

static ReferenceResult openssl_aes_gcm_reference(const std::vector<unsigned char>& plaintext,
                                 const std::array<unsigned char, 16>& key,
                                 const std::array<unsigned char, 12>& iv) {
    if (plaintext.size() > static_cast<size_t>(INT32_MAX)) {
        throw std::runtime_error("Payload exceeds EVP_EncryptUpdate's int-length limit in this example.");
    }

    ReferenceResult result;
    result.ciphertext.resize(plaintext.size() + AES_BLOCK_SIZE);
    int outLen = 0;
    int finalLen = 0;

    EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
    if (!ctx) throw std::runtime_error("EVP_CIPHER_CTX_new failed");

    bool ok = EVP_EncryptInit_ex(ctx, EVP_aes_128_gcm(), nullptr, nullptr, nullptr) == 1;
    ok = ok && EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_IVLEN,
                                  static_cast<int>(iv.size()), nullptr) == 1;
    ok = ok && EVP_EncryptInit_ex(ctx, nullptr, nullptr, key.data(), iv.data()) == 1;
    ok = ok && EVP_EncryptUpdate(ctx, result.ciphertext.data(), &outLen,
                                 plaintext.data(), static_cast<int>(plaintext.size())) == 1;
    ok = ok && EVP_EncryptFinal_ex(ctx, result.ciphertext.data() + outLen, &finalLen) == 1;
    ok = ok && EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_GET_TAG,
                                  static_cast<int>(result.tag.size()), result.tag.data()) == 1;
    EVP_CIPHER_CTX_free(ctx);

    if (!ok) throw std::runtime_error("OpenSSL AES-GCM encryption failed");
    if (static_cast<size_t>(outLen + finalLen) != plaintext.size()) {
        throw std::runtime_error("Unexpected ciphertext length from OpenSSL");
    }
    result.ciphertext.resize(plaintext.size());
    return result;
}

static double throughput_gib_s(size_t bytes, double latencyUs) {
    if (latencyUs <= 0.0) return 0.0;
    return (static_cast<double>(bytes) / (latencyUs * 1e-6)) / (1024.0 * 1024.0 * 1024.0);
}

int main(int argc, char** argv) {
    try {
        const bool smoke = (argc > 1 && std::string(argv[1]) == "--smoke");
        const int trials = smoke ? 1 : 50;
        // The smoke test deliberately exercises both a single reduction stage
        // and a multi-stage GHASH tree without running the largest payloads.
        std::vector<size_t> payloadSizes = smoke
            ? std::vector<size_t>{16 * 1024ULL, 8ULL * 1024 * 1024}
            : std::vector<size_t>{
                16 * 1024ULL,
                512 * 1024ULL,
                8ULL * 1024 * 1024,
                64ULL * 1024 * 1024,
                512ULL * 1024 * 1024,
                1ULL * 1024 * 1024 * 1024
              };

        std::mt19937 rng(std::random_device{}());
        std::ofstream csv("aes_gcm_gpu_only_results.csv");
        if (!csv) throw std::runtime_error("Could not open output CSV");
        csv << "PayloadSizeBytes,Trial,GPU_KernelPipeline_us,GPU_KernelPipeline_Throughput_GiB_s,"
               "GPU_EndToEnd_us,GPU_EndToEnd_Throughput_GiB_s,Validation\n";

        for (size_t payloadSize : payloadSizes) {
            if (payloadSize % (AES_BLOCK_SIZE * CHUNK_BLOCKS) != 0) {
                throw std::runtime_error("This GHASH tree version requires payload sizes divisible by 4096 bytes.");
            }
            if (payloadSize > static_cast<size_t>(INT32_MAX)) {
                throw std::runtime_error("Payload exceeds this example's OpenSSL int-length limit.");
            }

            const size_t numBlocks = payloadSize / AES_BLOCK_SIZE;
            const size_t numChunks = numBlocks / CHUNK_BLOCKS;
            const int threads = 256;
            const int ctrGrid = static_cast<int>((numBlocks + threads - 1) / threads);
            const int chunkGrid = static_cast<int>((numChunks + threads - 1) / threads);

            // Persistent buffers are allocated outside each timed trial.
            // End-to-end time below includes key expansion, H2D/D2H transfers and all GPU kernels,
            // but excludes cudaMalloc/cudaFree. Document this boundary in the EE.
            unsigned char *d_plain = nullptr, *d_cipher = nullptr, *d_key = nullptr, *d_roundKeys = nullptr;
            unsigned char *d_iv = nullptr, *d_H = nullptr, *d_tagMask = nullptr, *d_tag = nullptr;
            unsigned char *d_hashA = nullptr, *d_hashB = nullptr, *d_power = nullptr;
            CUDA_CHECK(cudaMalloc(&d_plain, payloadSize));
            CUDA_CHECK(cudaMalloc(&d_cipher, payloadSize));
            CUDA_CHECK(cudaMalloc(&d_key, 16));
            CUDA_CHECK(cudaMalloc(&d_roundKeys, 176));
            CUDA_CHECK(cudaMalloc(&d_iv, 12));
            CUDA_CHECK(cudaMalloc(&d_H, 16));
            CUDA_CHECK(cudaMalloc(&d_tagMask, 16));
            CUDA_CHECK(cudaMalloc(&d_tag, 16));
            CUDA_CHECK(cudaMalloc(&d_hashA, numChunks * 16));
            CUDA_CHECK(cudaMalloc(&d_hashB, numChunks * 16));
            CUDA_CHECK(cudaMalloc(&d_power, 16));

            cudaEvent_t kernelStart, kernelStop;
            CUDA_CHECK(cudaEventCreate(&kernelStart));
            CUDA_CHECK(cudaEventCreate(&kernelStop));

            std::cout << "Payload " << payloadSize << " bytes: running " << trials << " trial(s)...\n";

            for (int trial = 1; trial <= trials; ++trial) {
                std::vector<unsigned char> plaintext(payloadSize);
                for (auto& b : plaintext) b = static_cast<unsigned char>(rng() & 0xff);

                std::array<unsigned char, 16> key{};
                std::array<unsigned char, 12> iv{};
                for (auto& b : key) b = static_cast<unsigned char>(rng() & 0xff);
                for (auto& b : iv) b = static_cast<unsigned char>(rng() & 0xff);

                // Generate a correctness reference with OpenSSL. This is NOT a CPU benchmark:
                // no CPU timing is recorded, and this call occurs before either GPU timer starts.
                ReferenceResult reference = openssl_aes_gcm_reference(plaintext, key, iv);

                std::vector<unsigned char> gpuCiphertext(payloadSize);
                std::array<unsigned char, 16> gpuTag{};
                const auto e2eStart = std::chrono::steady_clock::now();
                CUDA_CHECK(cudaMemcpy(d_key, key.data(), key.size(), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_iv, iv.data(), iv.size(), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_plain, plaintext.data(), payloadSize, cudaMemcpyHostToDevice));

                CUDA_CHECK(cudaEventRecord(kernelStart));
                aes128_key_expansion_kernel<<<1, 1>>>(d_key, d_roundKeys);
                CUDA_CHECK(cudaGetLastError());

                derive_gcm_values_kernel<<<1, 1>>>(d_roundKeys, d_iv, d_H, d_tagMask);
                CUDA_CHECK(cudaGetLastError());

                aes_ctr_kernel<<<ctrGrid, threads>>>(d_plain, d_cipher, d_roundKeys, d_iv, numBlocks);
                CUDA_CHECK(cudaGetLastError());

                ghash_chunks_kernel<<<chunkGrid, threads>>>(d_cipher, d_H, d_hashA, numBlocks);
                CUDA_CHECK(cudaGetLastError());

                size_t currentCount = numChunks;
                int level = 1;
                unsigned char* current = d_hashA;
                unsigned char* next = d_hashB;

                while (currentCount > 1) {
                    const size_t outputCount = (currentCount + REDUCE_FANIN - 1) / REDUCE_FANIN;
                    compute_h_power_kernel<<<1, 1>>>(d_H, d_power, level * 8);
                    CUDA_CHECK(cudaGetLastError());
                    const int reduceGrid = static_cast<int>((outputCount + threads - 1) / threads);
                    reduce_ghash_kernel<<<reduceGrid, threads>>>(current, next, d_power, currentCount);
                    CUDA_CHECK(cudaGetLastError());

                    currentCount = outputCount;
                    unsigned char* oldCurrent = current;
                    current = next;
                    next = oldCurrent;
                    ++level;
                }

                finish_gcm_tag_kernel<<<1, 1>>>(current, d_H, d_tagMask,
                                               static_cast<uint64_t>(payloadSize), d_tag);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaEventRecord(kernelStop));
                CUDA_CHECK(cudaEventSynchronize(kernelStop));

                float kernelMs = 0.0f;
                CUDA_CHECK(cudaEventElapsedTime(&kernelMs, kernelStart, kernelStop));
                const double gpuKernelUs = static_cast<double>(kernelMs) * 1000.0;

                CUDA_CHECK(cudaMemcpy(gpuCiphertext.data(), d_cipher, payloadSize, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(gpuTag.data(), d_tag, gpuTag.size(), cudaMemcpyDeviceToHost));
                const auto e2eEnd = std::chrono::steady_clock::now();
                const double gpuE2eUs = std::chrono::duration<double, std::micro>(e2eEnd - e2eStart).count();

                const bool ciphertextMatches = (gpuCiphertext == reference.ciphertext);
                const bool tagMatches = (gpuTag == reference.tag);
                const bool valid = ciphertextMatches && tagMatches;
                if (!valid) {
                    std::cerr << "Validation failed for payload=" << payloadSize << ", trial=" << trial
                              << ". Ciphertext match=" << ciphertextMatches
                              << ", tag match=" << tagMatches << "\n";
                    throw std::runtime_error("GPU result did not match OpenSSL AES-GCM. Do not use these benchmark results.");
                }

                const double kernelThr = throughput_gib_s(payloadSize, gpuKernelUs);
                const double e2eThr = throughput_gib_s(payloadSize, gpuE2eUs);
                csv << payloadSize << ',' << trial << ',' << gpuKernelUs << ',' << kernelThr << ','
                    << gpuE2eUs << ',' << e2eThr << ",PASS\n";
                csv.flush();

                std::cout << "  Trial " << trial << ": validation PASS; GPU kernel pipeline " << gpuKernelUs
                          << " us, GPU end-to-end " << gpuE2eUs << " us\n";
            }

            CUDA_CHECK(cudaEventDestroy(kernelStart));
            CUDA_CHECK(cudaEventDestroy(kernelStop));
            CUDA_CHECK(cudaFree(d_plain));
            CUDA_CHECK(cudaFree(d_cipher));
            CUDA_CHECK(cudaFree(d_key));
            CUDA_CHECK(cudaFree(d_roundKeys));
            CUDA_CHECK(cudaFree(d_iv));
            CUDA_CHECK(cudaFree(d_H));
            CUDA_CHECK(cudaFree(d_tagMask));
            CUDA_CHECK(cudaFree(d_tag));
            CUDA_CHECK(cudaFree(d_hashA));
            CUDA_CHECK(cudaFree(d_hashB));
            CUDA_CHECK(cudaFree(d_power));
        }

        csv.close();
        std::cout << "Finished. Results written to aes_gcm_gpu_only_results.csv\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "ERROR: " << ex.what() << "\n";
        return 1;
    }
}
