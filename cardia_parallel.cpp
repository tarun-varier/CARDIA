#include <bits/stdc++.h>
#include <omp.h>
using namespace std;
using hrc_t = chrono::high_resolution_clock;

// ---------------- CSV loader ----------------
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
        if (pEnd != line.c_str()) x.push_back(val);
    }
    return x;
}

// ---------------- simple bandpass (global) ----------------
vector<double> simple_bandpass(const vector<double>& x, int fs) {
    int N = (int)x.size();
    vector<double> y(N, 0.0);
    int hp_win = max(1, int(0.6 * fs));        // highpass moving-average window
    vector<double> ma(N, 0.0);
    double sum = 0.0;
    for (int i = 0; i < N; ++i) {
        sum += x[i];
        if (i >= hp_win) sum -= x[i - hp_win];
        ma[i] = sum / min(i + 1, hp_win);
    }
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) y[i] = x[i] - ma[i];

    int lp_win = max(1, int(0.12 * fs));       // lowpass moving-average window
    vector<double> out(N, 0.0);
    sum = 0.0;
    for (int i = 0; i < N; ++i) {
        sum += y[i];
        if (i >= lp_win) sum -= y[i - lp_win];
        out[i] = sum / min(i + 1, lp_win);
    }
    return out;
}

// ---------------- moving window integration ----------------
vector<double> moving_window_integral(const vector<double>& x, int window_samples) {
    int N = (int)x.size();
    vector<double> out(N, 0.0);
    double sum = 0.0;
    for (int i = 0; i < N; ++i) {
        sum += x[i];
        if (i >= window_samples) sum -= x[i - window_samples];
        if (window_samples > 0) out[i] = sum / (double)window_samples;
    }
    return out;
}

vector<int> detect_peaks_local(const vector<double>& filtered, const vector<double>& integ, int fs) {
    int N = (int)filtered.size();
    vector<int> peaks;

    // Robust mean/std (parallel reductions)
    double mean = 0.0;
    #pragma omp parallel for reduction(+:mean)
    for (int i = 0; i < N; ++i) mean += integ[i];
    mean /= max(1, N);

    double var = 0.0;
    #pragma omp parallel for reduction(+:var)
    for (int i = 0; i < N; ++i) var += (integ[i] - mean) * (integ[i] - mean);
    double sigma = sqrt(var / max(1, N));

    double threshold = mean + 1.5 * sigma;
    int refractory = max(1, int(0.2 * fs)); // samples

    for (int i = 1; i < N - 1; ++i) {
        if (integ[i] > threshold && integ[i] > integ[i - 1] && integ[i] >= integ[i + 1]) {
            int win = max(1, int(0.04 * fs));
            int lo = max(0, i - win);
            int hi = min(N - 1, i + win);
            int peak_idx = lo;
            double peak_val = filtered[lo];
            for (int j = lo + 1; j <= hi; ++j) {
                if (filtered[j] > peak_val) {
                    peak_val = filtered[j];
                    peak_idx = j;
                }
            }
            if (peaks.empty() || peak_idx - peaks.back() > refractory)
                peaks.push_back(peak_idx);
        }
    }
    return peaks;
}

// ---------------- chunk process ----------------
struct ChunkResult {
    int chunk_index;
    int start_global;            // start index in full filtered array that corresponds to chunk[0]
    vector<int> peaks_global;    // peak indices in global coordinates
    double elapsed_ms;
};

ChunkResult process_chunk_task(const vector<double>& filtered_full,
                               int fs,
                               int actual_start,
                               int actual_end,
                               int chunk_index,
                               int mwin_samples) {
    auto t0 = hrc_t::now();

    int N = actual_end - actual_start;
    vector<double> filtered_chunk(N);
    for (int i = 0; i < N; ++i) filtered_chunk[i] = filtered_full[actual_start + i];

    vector<double> squared(N, 0.0);
    #pragma omp parallel for schedule(static)
    for (int i = 1; i < N; ++i) {
        double d = filtered_chunk[i] - filtered_chunk[i - 1];
        squared[i] = d * d;
    }
    vector<double> integ = moving_window_integral(squared, mwin_samples);

    vector<int> peaks_local = detect_peaks_local(filtered_chunk, integ, fs);

    vector<int> peaks_global;
    peaks_global.reserve(peaks_local.size());
    for (int p : peaks_local) peaks_global.push_back(p + actual_start);

    auto t1 = hrc_t::now();
    double elapsed_ms = chrono::duration<double, milli>(t1 - t0).count();

    return {chunk_index, actual_start, peaks_global, elapsed_ms};
}

// ---------------- merge peaks (deterministic) ----------------
vector<int> merge_and_resolve_peaks(const vector<int>& all_peaks_sorted,
                                    const vector<double>& filtered_full,
                                    int fs) {
    if (all_peaks_sorted.empty()) return {};

    int refractory = max(1, int(0.2 * fs));
    vector<int> final_peaks;
    final_peaks.reserve(all_peaks_sorted.size());

    for (int p : all_peaks_sorted) {
        if (final_peaks.empty()) {
            final_peaks.push_back(p);
            continue;
        }
        int last = final_peaks.back();
        if (p - last > refractory) {
            final_peaks.push_back(p);
        } else {
            if (filtered_full[p] > filtered_full[last]) {
                final_peaks.back() = p;
            }
        }
    }
    return final_peaks;
}

// ---------------- main ----------------
int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <path_to_ecg_csv>\n";
        return 1;
    }
    string filepath = argv[1];
    int fs = 360;

    auto ecg = load_csv_ecg(filepath);
    int N = (int)ecg.size();
    if (N == 0) {
        cerr << "No samples loaded.\n";
        return 1;
    }

    cout << "Loaded " << N << " samples from " << filepath << "\n";

    auto tfilter0 = hrc_t::now();
    vector<double> filtered_full = simple_bandpass(ecg, fs);
    auto tfilter1 = hrc_t::now();
    double filter_time_ms = chrono::duration<double, milli>(tfilter1 - tfilter0).count();
    cout << "Bandpass (global) done in " << fixed << setprecision(2) << filter_time_ms << " ms\n";

    int mwin_samples = max(1, int(0.15 * fs));   // same as used in previous code

    // chunking parameters
    int chunk_seconds = 5;
    int chunk_samples = chunk_seconds * fs;
    int min_overlap = max(mwin_samples, int(0.5 * fs));
    int overlap = min_overlap;
    int step = max(1, chunk_samples - overlap);
    int num_chunks = (N + step - 1) / step;

    cout << "Chunk size: " << chunk_samples << " samples, overlap: " << overlap
         << " samples, step: " << step << ", num_chunks: " << num_chunks << "\n";

    vector<ChunkResult> chunk_results(num_chunks);

    auto T0 = hrc_t::now();

    #pragma omp parallel
    {
        #pragma omp single
        {
            for (int c = 0; c < num_chunks; ++c) {
                int nominal_start = c * step;
                int nominal_end = min(N, nominal_start + chunk_samples);

                int actual_start = max(0, nominal_start - mwin_samples);
                int actual_end = min(N, nominal_end + mwin_samples);

                #pragma omp task firstprivate(c, actual_start, actual_end)
                {
                    chunk_results[c] = process_chunk_task(filtered_full, fs, actual_start, actual_end, c, mwin_samples);
                }
            }
        }
    } // implicit barrier to wait for tasks

    auto T1 = hrc_t::now();
    double total_proc_time_ms = chrono::duration<double, milli>(T1 - T0).count();

    vector<int> all_peaks;
    for (const auto& cr : chunk_results) {
        for (int p : cr.peaks_global) all_peaks.push_back(p);
    }

    sort(all_peaks.begin(), all_peaks.end());
    all_peaks.erase(unique(all_peaks.begin(), all_peaks.end()), all_peaks.end());

    vector<int> final_peaks = merge_and_resolve_peaks(all_peaks, filtered_full, fs);

    double avg_bpm = 0.0;
    if (final_peaks.size() > 1) {
        vector<double> rr(final_peaks.size() - 1);
        for (size_t i = 1; i < final_peaks.size(); ++i)
            rr[i - 1] = (final_peaks[i] - final_peaks[i - 1]) / double(fs);
        double mean_rr = accumulate(rr.begin(), rr.end(), 0.0) / rr.size();
        if (mean_rr > 0.0) avg_bpm = 60.0 / mean_rr;
    }

    cout << "\n--- RESULTS ---\n";
    cout << "Total samples: " << N << "\n";
    cout << "Detected peaks (after merge): " << final_peaks.size() << "\n";
    cout << "Avg BPM: " << fixed << setprecision(2) << avg_bpm << "\n";
    cout << "Filtering time (global): " << fixed << setprecision(2) << filter_time_ms << " ms\n";
    cout << "Chunked processing time (parallel tasks total): " << fixed << setprecision(2) << total_proc_time_ms << " ms\n";

    cout << "First 10 peaks: ";
    for (size_t i = 0; i < final_peaks.size() && i < 10; ++i) cout << final_peaks[i] << " ";
    cout << "\n";

    return 0;
}
