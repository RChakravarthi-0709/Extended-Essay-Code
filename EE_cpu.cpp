#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <fstream>
#include <openssl/evp.h>
#include <openssl/crypto.h>


int main() {
    std::vector<size_t> payloadSizes = {
        16 * 1024ULL,        // 16 KiB
        512 * 1024ULL,       // 512 KiB
        8ULL * 1024 * 1024,  // 8 MiB
        64ULL * 1024 * 1024, // 64 MiB
        512ULL * 1024 * 1024,// 512 MiB
        1ULL * 1024 * 1024 * 1024 // 1 GiB
    };




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

            std::vector<unsigned char> tag(16);

            int len = 0;
            int final_len = 0;

            EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();

            if (!ctx) {
                std::cerr << "Failed to create OpenSSL context\n";
                return 1;
            }

            auto start = std::chrono::steady_clock::now();

            bool ok =
                EVP_EncryptInit_ex(
                    ctx, EVP_aes_128_gcm(), nullptr,
                    key.data(), iv.data()
                ) == 1;

            if (ok) {
                ok = EVP_EncryptUpdate(
                    ctx, ciphertext.data(), &len,
                    plaintext.data(),
                    static_cast<int>(plaintext.size())
                ) == 1;
            }

            if (ok) {
                ok = EVP_EncryptFinal_ex(
                    ctx, ciphertext.data() + len, &final_len
                ) == 1;
            }

            if (ok) {
                ok = EVP_CIPHER_CTX_ctrl(
                    ctx, EVP_CTRL_GCM_GET_TAG, 16, tag.data()
                ) == 1;
            }

            auto end = std::chrono::steady_clock::now();

            EVP_CIPHER_CTX_free(ctx);

            if (!ok) {
                std::cerr << "AES-GCM encryption failed\n";
                return 1;
            }

            double latency_us =
                std::chrono::duration<double, std::micro>(
                    end - start
                ).count();

            double throughput =
                static_cast<double>(payloadSize) /
                (latency_us / 1e6);

            ciphertext.resize(len + final_len);


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
