#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <chrono>
#include <cmath>
#include <numeric>
#include <iomanip>
#include <algorithm>
#include <omp.h>

using namespace std;
using hrc_t = chrono::high_resolution_clock;

// ---------- 1. Optimized CSV loader ----------
vector<double> load_csv_ecg(const string& filename) {
    ifstream fin(filename);
    if (!fin.is_open()) {
        cerr << "Error: Cannot open " << filename << "\n";
        exit(1);
    }
    vector<double> x;
    x.reserve(2000000); 
    string line;
    while (getline(fin, line)) {
        if (line.empty()) continue;
        char* pEnd;
        double val = strtod(line.c_str(), &pEnd);
        if (pEnd != line.c_str()) { // Check if a conversion happened
            x.push_back(val);
        }
    }
    cout << "Loaded " << x.size() << " samples from " << filename << "\n";
    return x;
}

// ---------- ECG pipeline ----------
vector<double> simple_bandpass(const vector<double>& x, int fs) {
    int N = x.size();
    vector<double> y(N, 0.0);
    int hp_win = max(1, int(0.6 * fs));
    vector<double> ma(N, 0.0);
    double sum = 0.0;

    for (int i = 0; i < N; ++i) {
        sum += x[i];
        if (i >= hp_win) sum -= x[i - hp_win];
        ma[i] = sum / min(i + 1, hp_win);
    }

    // Highpass
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) y[i] = x[i] - ma[i];

    // Lowpass (also sequential)
    int lp_win = max(1, int(0.12 * fs));
    vector<double> out(N, 0.0);
    sum = 0.0;
    for (int i = 0; i < N; ++i) {
        sum += y[i];
        if (i >= lp_win) sum -= y[i - lp_win];
        out[i] = sum / min(i + 1, lp_win);
    }
    return out;
}


// Sequential moving window integral
vector<double> moving_window_integral(const vector<double>& x, int window_samples) {
    int N = x.size();
    vector<double> out(N, 0.0);
    double sum = 0.0;
    for (int i = 0; i < N; ++i) {
        sum += x[i];
        if (i >= window_samples) sum -= x[i - window_samples];
        if (window_samples > 0) {
            out[i] = sum / (double)window_samples;
        }
    }
    return out;
}

// ---------- Parallelized stats ----------
vector<int> detect_peaks(const vector<double>& raw, const vector<double>& integ, int fs) {
    int N = raw.size();
    vector<int> peaks;

    double mean = 0.0;
    #pragma omp parallel for reduction(+:mean)
    for (int i = 0; i < N; ++i)
        mean += integ[i];
    mean /= N;

    double var = 0.0;
    #pragma omp parallel for reduction(+:var)
    for (int i = 0; i < N; ++i)
        var += (integ[i] - mean) * (integ[i] - mean);
    double sigma = sqrt(var / N);

    double threshold = mean + 1.5 * sigma;
    int refractory = int(0.2 * fs);

    for (int i = 1; i < N - 1; ++i) {
        if (integ[i] > threshold && integ[i] > integ[i - 1] && integ[i] >= integ[i + 1]) {
            int win = max(1, int(0.04 * fs));
            int lo = max(0, i - win);
            int hi = min(N - 1, i + win);
            int peak_idx = lo;
            double peak_val = raw[lo];
            for (int j = lo + 1; j <= hi; ++j)
                if (raw[j] > peak_val) {
                    peak_val = raw[j];
                    peak_idx = j;
                }
            if (peaks.empty() || peak_idx - peaks.back() > refractory)
                peaks.push_back(peak_idx);
        }
    }
    return peaks;
}

// ---------- 2. Main computation with fused loops ----------
void compute_and_print_results(const vector<double>& signal, int fs) {
    int N = signal.size();
    auto t0 = hrc_t::now();

    auto filtered = simple_bandpass(signal, fs);
    
    vector<double> squared(N, 0.0);
    #pragma omp parallel for schedule(static)
    for (int i = 1; i < N; ++i) {
        double d = filtered[i] - filtered[i - 1];
        squared[i] = d * d;
    }

    int mwin_samples = max(1, int(0.15 * fs));
    auto integ = moving_window_integral(squared, mwin_samples);
    auto peaks = detect_peaks(filtered, integ, fs);

    auto t1 = hrc_t::now();
    double elapsed_ms = chrono::duration<double, milli>(t1 - t0).count();

    vector<double> rr;
    if (peaks.size() > 1) {
        rr.resize(peaks.size() - 1);
        #pragma omp parallel for schedule(static)
        for (size_t i = 1; i < peaks.size(); ++i)
            rr[i - 1] = (peaks[i] - peaks[i - 1]) / double(fs);
    }
    
    double avg_bpm = 0.0;
    if (!rr.empty()) {
        double mean_rr = accumulate(rr.begin(), rr.end(), 0.0) / rr.size();
        avg_bpm = 60.0 / mean_rr;
    }

    cout << "Samples: " << N
         << ", Beats: " << peaks.size()
         << ", Avg BPM: " << fixed << setprecision(2) << avg_bpm
         << ", Time: " << fixed << setprecision(3) << elapsed_ms << " ms\n";
    cout << "R-peaks (first 10):";
    for (size_t i = 0; i < peaks.size() && i < 10; ++i)
        cout << " " << peaks[i];
    cout << "\n\n";
}


// ---------- main ----------
int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <path_to_ecg_csv>\n";
        cerr << "Example: ./ecg_realdata C:/Users/SAHEER/OneDrive/Desktop/college/ecg100.csv\n";
        return 1;
    }
    string filepath = argv[1];

    int fs = 360;

    auto ecg_full = load_csv_ecg(filepath);
    int N = ecg_full.size();
    cout << "Total loaded samples: " << N << "\n\n";

    vector<int> test_sizes = {6500, 65000, N};

    for (int size : test_sizes) {
        if (size > N) {
            cout << "Skipping size " << size << " (exceeds dataset length)\n";
            continue;
        }
        cout << "=== Running for first " << size << " samples ===\n";
        vector<double> ecg_sub(ecg_full.begin(), ecg_full.begin() + size);
        compute_and_print_results(ecg_sub, fs);
    }

    return 0;
}
