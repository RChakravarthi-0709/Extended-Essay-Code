#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <fstream>
#include <openssl/evp.h>
#include <openssl/crypto.h>

int main()
{
    // Initialise OpenSSL algorithms
    OPENSSL_init_crypto(0, nullptr);

    // Payload sizes to test (in bytes)
    const std::vector<size_t> payloadSizes = {
        16ULL * 1024ULL,                         // 16 KiB
        512ULL * 1024ULL,                        // 512 KiB
        8ULL * 1024ULL * 1024ULL,                // 8 MiB
        64ULL * 1024ULL * 1024ULL,               // 64 MiB
        512ULL * 1024ULL * 1024ULL,              // 512 MiB
        1024ULL * 1024ULL * 1024ULL              // 1 GiB
    };

    const int trialsPerSize = 50;

    // Random number generation using std::mt19937 seeded with std::random_device
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> dist(0, 255);

    // Open CSV output file
    std::ofstream csvFile("cpu_results.csv");
    if (!csvFile.is_open())
    {
        std::cerr << "Failed to create cpu_results.csv" << std::endl;
        return 1;
    }

    // Write CSV header
    csvFile << "PayloadSize(Bytes),Trial,Latency(us),Throughput(Bps),Mode\n";

    // Benchmark each payload size
    for (size_t payloadSize : payloadSizes)
    {
        double totalLatencyUs = 0.0;
        double totalThroughput = 0.0;

        std::cout << "Testing payload size: "
                  << payloadSize
                  << " bytes"
                  << std::endl;

        for (int trial = 1; trial <= trialsPerSize; ++trial)
        {
            // Generate random AES-128 key (16 bytes)
            std::vector<unsigned char> key(16);
            for (auto &b : key)
            {
                b = static_cast<unsigned char>(dist(rng));
            }

            // Generate random GCM IV (12 bytes)
            std::vector<unsigned char> iv(12);
            for (auto &b : iv)
            {
                b = static_cast<unsigned char>(dist(rng));
            }

            // Generate random plaintext
            std::vector<unsigned char> plaintext(payloadSize);
            for (auto &b : plaintext)
            {
                b = static_cast<unsigned char>(dist(rng));
            }

            // Allocate ciphertext buffer as payloadSize + 16 bytes
            std::vector<unsigned char> ciphertext(payloadSize + 16);

            int outLen = 0;

            // Create encryption context
            EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
            if (!ctx)
            {
                std::cerr << "Failed to create EVP_CIPHER_CTX" << std::endl;
                return 1;
            }

            // Measure only the encryption stage
            auto startTime = std::chrono::high_resolution_clock::now();

            // Initialise AES-128-GCM encryption
            if (EVP_EncryptInit_ex(ctx, EVP_aes_128_gcm(), nullptr, nullptr, nullptr) != 1)
            {
                std::cerr << "EVP_EncryptInit_ex failed" << std::endl;
                EVP_CIPHER_CTX_free(ctx);
                return 1;
            }

            if (EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_IVLEN,
                                    static_cast<int>(iv.size()), nullptr) != 1)
            {
                std::cerr << "Failed to set IV length" << std::endl;
                EVP_CIPHER_CTX_free(ctx);
                return 1;
            }

            if (EVP_EncryptInit_ex(ctx, nullptr, nullptr,
                                   key.data(), iv.data()) != 1)
            {
                std::cerr << "Failed to set key and IV" << std::endl;
                EVP_CIPHER_CTX_free(ctx);
                return 1;
            }

            // Encrypt plaintext
            if (EVP_EncryptUpdate(ctx,
                                  ciphertext.data(),
                                  &outLen,
                                  plaintext.data(),
                                  static_cast<int>(payloadSize)) != 1)
            {
                std::cerr << "EVP_EncryptUpdate failed" << std::endl;
                EVP_CIPHER_CTX_free(ctx);
                return 1;
            }

            auto endTime = std::chrono::high_resolution_clock::now();

            // Free context after encryption
            EVP_CIPHER_CTX_free(ctx);

            // Calculate latency in microseconds
            double latencyUs =
                std::chrono::duration<double, std::micro>(endTime - startTime).count();

            // Calculate throughput in bytes per second
            double throughputBps =
                static_cast<double>(payloadSize) / (latencyUs / 1e6);

            totalLatencyUs += latencyUs;
            totalThroughput += throughputBps;

            // Record individual trial result
            csvFile << payloadSize << ","
                    << trial << ","
                    << latencyUs << ","
                    << throughputBps << ","
                    << "CPU\n";
        }

        // Calculate averages after 50 trials
        double averageLatencyUs = totalLatencyUs / static_cast<double>(trialsPerSize);
        double averageThroughput = totalThroughput / static_cast<double>(trialsPerSize);

        // Write average row
        csvFile << payloadSize << ","
                << "Average,"
                << averageLatencyUs << ","
                << averageThroughput << ","
                << "CPU\n";

        // Print averages to console
        std::cout << "Average Latency (us): "
                  << averageLatencyUs
                  << std::endl;

        std::cout << "Average Throughput (Bps): "
                  << averageThroughput
                  << std::endl;

        std::cout << "----------------------------------------"
                  << std::endl;
    }

    csvFile.close();

    std::cout << "Benchmark complete. Results saved to cpu_results.csv"
              << std::endl;

    return 0;
}
