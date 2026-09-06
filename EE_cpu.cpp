#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <fstream>
#include <openssl/evp.h>
#include <openssl/crypto.h>

// Utility: check if AES-NI is supported
bool aesni_supported() {
    // OPENSSL_ia32cap_P is defined by OpenSSL to expose CPU feature bits
    extern unsigned long long OPENSSL_ia32cap_P[];
    unsigned long long caps = OPENSSL_ia32cap_P[1];
    return (caps & (1ULL << 57)); // Bit 57 = AES-NI
}

int main() {
    std::vector<size_t> payloadSizes = {
        16 * 1024ULL,        // 16 KiB
        512 * 1024ULL,       // 512 KiB
        8ULL * 1024 * 1024,  // 8 MiB
        64ULL * 1024 * 1024, // 64 MiB
        512ULL * 1024 * 1024,// 512 MiB
        1ULL * 1024 * 1024 * 1024 // 1 GiB
    };

    if (aesni_supported()) {
        std::cout << "AES-NI is supported and OpenSSL will use hardware acceleration.\n";
    } else {
        std::cout << "AES-NI not available — OpenSSL will fall back to software AES.\n";
    }

    std::mt19937 rng(std::random_device{}());
    std::ofstream csv("cpu_results.csv");
    csv << "PayloadSize(Bytes),Trial,Latency(us),Throughput(Bps),Mode\n";

    for (size_t payloadSize : payloadSizes) {
        std::vector<double> latencies, throughputs;

        for (int trial = 1; trial <= 50; ++trial) {
            // Random key (16 bytes) and IV (12 bytes)
            std::vector<unsigned char> key(16), iv(12);
            for (auto &b : key) b = rng() & 0xFF;
            for (auto &b : iv) b = rng() & 0xFF;

            // Random plaintext
            std::vector<unsigned char> plaintext(payloadSize);
            for (auto &b : plaintext) b = rng() & 0xFF;

            std::vector<unsigned char> ciphertext(payloadSize + 16);
            int len;

            auto start = std::chrono::high_resolution_clock::now();

            EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
            EVP_EncryptInit_ex(ctx, EVP_aes_128_gcm(), nullptr, key.data(), iv.data());
            EVP_EncryptUpdate(ctx, ciphertext.data(), &len, plaintext.data(), plaintext.size());
            EVP_CIPHER_CTX_free(ctx);

            auto end = std::chrono::high_resolution_clock::now();
            auto latency_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            double throughput = static_cast<double>(payloadSize) / (latency_us / 1e6);

            latencies.push_back(latency_us);
            throughputs.push_back(throughput);

            csv << payloadSize << "," << trial << "," << latency_us << "," << throughput << ",CPU\n";
        }

        double avgLatency = 0, avgThroughput = 0;
        for (int i = 0; i < 50; ++i) {
            avgLatency += latencies[i];
            avgThroughput += throughputs[i];
        }
        avgLatency /= 50.0;
        avgThroughput /= 50.0;

        csv << payloadSize << ",Average," << avgLatency << "," << avgThroughput << ",CPU\n";
        std::cout << "Payload " << payloadSize << " bytes: Avg Latency = "
                  << avgLatency << " us, Avg Throughput = "
                  << avgThroughput << " B/s\n";
    }

    csv.close();
    return 0;
}
