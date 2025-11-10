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

// ---------------- Bandpass Filter (5-18 Hz) - Parallelized ----------------
vector<double> bandpass_filter(const vector<double>& x, int fs) {
    int N = (int)x.size();
    vector<double> y(N, 0.0);
    
    // Low-pass component (sequential due to IIR dependency)
    double alpha_lp = 0.15;
    for (int i = 1; i < N; i++) {
        y[i] = alpha_lp * x[i] + (1 - alpha_lp) * y[i-1];
    }
    
    // High-pass component
    vector<double> lp_low(N, 0.0);
    double alpha_hp = 0.05;
    for (int i = 1; i < N; i++) {
        lp_low[i] = alpha_hp * y[i] + (1 - alpha_hp) * lp_low[i-1];
    }
    
    // Parallel subtraction
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        y[i] = y[i] - lp_low[i];
    }
    
    return y;
}

// ---------------- Flattop Window ----------------
vector<double> flattop_window(int Z) {
    vector<double> w(Z);
    double a0 = 0.21557895;
    double a1 = 0.41663158;
    double a2 = 0.27726316;
    double a3 = 0.08357895;
    double a4 = 0.00694737;
    
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < Z; n++) {
        double beta = 2.0 * M_PI * n / Z;
        w[n] = a0 - a1 * cos(beta) + a2 * cos(2*beta) 
               - a3 * cos(3*beta) + a4 * cos(4*beta);
    }
    return w;
}

// ---------------- Moving Average with Window ----------------
vector<double> moving_average_window(const vector<double>& x, int window_size) {
    int N = (int)x.size();
    vector<double> y(N, 0.0);
    
    auto window = flattop_window(window_size);
    
    // Sequential due to dependency, but can be chunked
    for (int i = 0; i < N; i++) {
        double sum = 0.0;
        double weight_sum = 0.0;
        for (int j = 0; j < window_size && (i - j) >= 0; j++) {
            sum += x[i - j] * window[j];
            weight_sum += window[j];
        }
        if (weight_sum > 0) y[i] = sum / weight_sum;
    }
    return y;
}

// ---------------- Derivative - Parallelized ----------------
vector<double> derivative(const vector<double>& x, int fs) {
    int N = (int)x.size();
    vector<double> y(N, 0.0);
    double T = 1.0 / fs;
    
    #pragma omp parallel for schedule(static)
    for (int n = 2; n < N - 2; n++) {
        y[n] = (1.0 / (8.0 * T)) * (-x[n-2] - 2*x[n-1] + 2*x[n+1] + x[n+2]);
    }
    return y;
}

// ---------------- Savitzky-Golay Filter - Parallelized ----------------
vector<double> savitzky_golay_filter(const vector<double>& x) {
    int N = (int)x.size();
    vector<double> y(N, 0.0);
    
    vector<double> coeff = {-36, 9, 44, 69, 84, 89, 84, 69, 44, 9, -36};
    double norm = 429.0;
    
    #pragma omp parallel for schedule(static)
    for (int i = 5; i < N - 5; i++) {
        double sum = 0.0;
        for (int j = -5; j <= 5; j++) {
            sum += coeff[j + 5] * x[i + j];
        }
        y[i] = sum / norm;
    }
    
    // Copy edges
    for (int i = 0; i < 5; i++) y[i] = x[i];
    for (int i = N - 5; i < N; i++) y[i] = x[i];
    
    return y;
}

// ---------------- Squaring - Parallelized ----------------
vector<double> squaring(const vector<double>& x) {
    int N = (int)x.size();
    vector<double> y(N);
    
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        y[i] = x[i] * x[i];
    }
    return y;
}

// ---------------- Moving Window Integration ----------------
vector<double> moving_window_integral(const vector<double>& x, int window_samples) {
    int N = (int)x.size();
    vector<double> out(N, 0.0);
    double sum = 0.0;
    
    for (int i = 0; i < N; i++) {
        sum += x[i];
        if (i >= window_samples) sum -= x[i - window_samples];
        if (window_samples > 0) out[i] = sum / (double)window_samples;
    }
    return out;
}

// ---------------- Improved Pan-Tompkins QRS Detection (Chunk) ----------------
struct ChunkResult {
    int chunk_index;
    int start_global;
    vector<int> peaks_global;
    double elapsed_ms;
};

ChunkResult detect_qrs_chunk(const vector<double>& integrated_chunk,
                             const vector<double>& filtered_chunk,
                             int actual_start,
                             int chunk_index,
                             int fs) {
    auto t0 = hrc_t::now();
    
    int N = (int)integrated_chunk.size();
    vector<int> peaks_local;
    
    // Initialize thresholds
    int init_samples = min(2 * fs, N);
    double max_h = *max_element(integrated_chunk.begin(), 
                                integrated_chunk.begin() + init_samples);
    double mean_h = accumulate(integrated_chunk.begin(), 
                               integrated_chunk.begin() + init_samples, 0.0) / init_samples;
    
    double THR1 = max_h / 3.0;
    double THR2 = 0.5 * mean_h;
    double THR3 = 0.5 * THR2;
    double ESP = THR1;
    double ENP = THR2;
    
    int min_interval = int(0.231 * fs);
    int last_peak = -min_interval;
    
    deque<int> rr_intervals;
    
    // First pass detection
    for (int i = 1; i < N - 1; i++) {
        if (integrated_chunk[i] > integrated_chunk[i-1] && 
            integrated_chunk[i] >= integrated_chunk[i+1]) {
            
            if (integrated_chunk[i] > THR1 && (i - last_peak) > min_interval) {
                bool valid = true;
                
                if (!peaks_local.empty() && (i - last_peak) < int(0.36 * fs)) {
                    int window = int(0.07 * fs);
                    int prev_peak = peaks_local.back();
                    
                    double curr_slope = 0.0, prev_slope = 0.0;
                    for (int j = max(0, i - window); j < min(N, i + window); j++) {
                        if (j > 0) curr_slope = max(curr_slope, abs(filtered_chunk[j] - filtered_chunk[j-1]));
                    }
                    for (int j = max(0, prev_peak - window); j < min(N, prev_peak + window); j++) {
                        if (j > 0) prev_slope = max(prev_slope, abs(filtered_chunk[j] - filtered_chunk[j-1]));
                    }
                    
                    if (curr_slope < 0.6 * prev_slope) {
                        valid = false;
                    }
                }
                
                if (valid) {
                    peaks_local.push_back(i);
                    
                    if (!rr_intervals.empty()) {
                        rr_intervals.push_back(i - last_peak);
                        if (rr_intervals.size() > 8) rr_intervals.pop_front();
                    } else {
                        rr_intervals.push_back(i - last_peak);
                    }
                    
                    last_peak = i;
                    ESP = 0.75 * integrated_chunk[i] + 0.25 * ESP;
                    THR1 = ENP + 0.25 * (ESP - ENP);
                    THR2 = 0.4 * THR1;
                }
            }
        }
    }
    
    // Search-back
    vector<int> additional_peaks;
    double mean_rr = 0.0;
    if (!rr_intervals.empty()) {
        mean_rr = accumulate(rr_intervals.begin(), rr_intervals.end(), 0.0) / rr_intervals.size();
    }
    
    for (size_t k = 0; k < peaks_local.size() - 1; k++) {
        int gap = peaks_local[k+1] - peaks_local[k];
        
        if (gap > fs || (mean_rr > 0 && gap > 1.66 * mean_rr)) {
            int search_start = peaks_local[k] + int(0.36 * fs);
            int search_end = peaks_local[k+1];
            
            for (int i = search_start; i < search_end; i++) {
                if (i > 0 && i < N-1 && 
                    integrated_chunk[i] > integrated_chunk[i-1] && 
                    integrated_chunk[i] >= integrated_chunk[i+1] &&
                    integrated_chunk[i] > THR3) {
                    additional_peaks.push_back(i);
                    ESP = 0.75 * integrated_chunk[i] + 0.25 * ESP;
                    ENP = 0.75 * integrated_chunk[i] + 0.25 * ENP;
                    break;
                }
            }
        }
    }
    
    peaks_local.insert(peaks_local.end(), additional_peaks.begin(), additional_peaks.end());
    sort(peaks_local.begin(), peaks_local.end());
    
    // Convert to global coordinates
    vector<int> peaks_global;
    peaks_global.reserve(peaks_local.size());
    for (int p : peaks_local) {
        peaks_global.push_back(p + actual_start);
    }
    
    auto t1 = hrc_t::now();
    double elapsed_ms = chrono::duration<double, milli>(t1 - t0).count();
    
    return {chunk_index, actual_start, peaks_global, elapsed_ms};
}

// ---------------- Merge peaks ----------------
vector<int> merge_and_resolve_peaks(const vector<int>& all_peaks_sorted,
                                    const vector<double>& integrated,
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
            if (integrated[p] > integrated[last]) {
                final_peaks.back() = p;
            }
        }
    }
    return final_peaks;
}

// ---------------- Main ----------------
int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <path_to_ecg_csv>\n";
        return 1;
    }
    
    string filepath = argv[1];
    int fs = 360;
    
    cout << "Loading ECG data from " << filepath << "...\n";
    auto ecg = load_csv_ecg(filepath);
    int N = (int)ecg.size();
    
    if (N == 0) {
        cerr << "No samples loaded.\n";
        return 1;
    }
    
    cout << "Loaded " << N << " samples\n";
    
    auto T0 = hrc_t::now();
    
    // Step 1: Bandpass filter (5-18 Hz)
    cout << "Applying bandpass filter (5-18 Hz)...\n";
    auto tfilter0 = hrc_t::now();
    auto filtered = bandpass_filter(ecg, fs);
    auto tfilter1 = hrc_t::now();
    double filter_time_ms = chrono::duration<double, milli>(tfilter1 - tfilter0).count();
    
    // Step 2: Derivative
    cout << "Computing derivative...\n";
    auto diff = derivative(filtered, fs);
    
    // Step 3: Savitzky-Golay smoothing
    cout << "Applying Savitzky-Golay filter...\n";
    auto smoothed = savitzky_golay_filter(diff);
    
    // Step 4: Moving average with flattop window (60ms)
    cout << "Applying moving average with flattop window...\n";
    int ma_window = int(0.06 * fs);
    auto ma_smoothed = moving_average_window(smoothed, ma_window);
    
    // Step 5: Squaring
    cout << "Squaring signal...\n";
    auto squared = squaring(ma_smoothed);
    
    // Step 6: Moving window integration (150ms)
    cout << "Moving window integration...\n";
    int mwin_samples = int(0.15 * fs);
    auto integrated = moving_window_integral(squared, mwin_samples);
    
    auto tpreproc1 = hrc_t::now();
    double preproc_time_ms = chrono::duration<double, milli>(tpreproc1 - T0).count();
    
    // Step 7: Parallel chunked QRS detection
    cout << "Detecting QRS complexes with parallel improved Pan-Tompkins...\n";
    
    int chunk_seconds = 5;
    int chunk_samples = chunk_seconds * fs;
    int min_overlap = max(mwin_samples, int(0.5 * fs));
    int overlap = min_overlap;
    int step = max(1, chunk_samples - overlap);
    int num_chunks = (N + step - 1) / step;
    
    cout << "Chunk size: " << chunk_samples << " samples, overlap: " << overlap
         << " samples, step: " << step << ", num_chunks: " << num_chunks << "\n";
    
    vector<ChunkResult> chunk_results(num_chunks);
    
    auto tdetect0 = hrc_t::now();
    
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
                    int chunk_size = actual_end - actual_start;
                    vector<double> integrated_chunk(chunk_size);
                    vector<double> filtered_chunk(chunk_size);
                    
                    for (int i = 0; i < chunk_size; i++) {
                        integrated_chunk[i] = integrated[actual_start + i];
                        filtered_chunk[i] = filtered[actual_start + i];
                    }
                    
                    chunk_results[c] = detect_qrs_chunk(integrated_chunk, filtered_chunk,
                                                        actual_start, c, fs);
                }
            }
        }
    }
    
    auto tdetect1 = hrc_t::now();
    double detect_time_ms = chrono::duration<double, milli>(tdetect1 - tdetect0).count();
    
    // Merge peaks
    vector<int> all_peaks;
    for (const auto& cr : chunk_results) {
        for (int p : cr.peaks_global) {
            all_peaks.push_back(p);
        }
    }
    
    sort(all_peaks.begin(), all_peaks.end());
    all_peaks.erase(unique(all_peaks.begin(), all_peaks.end()), all_peaks.end());
    
    vector<int> final_peaks = merge_and_resolve_peaks(all_peaks, integrated, fs);
    
    auto T1 = hrc_t::now();
    double total_time_ms = chrono::duration<double, milli>(T1 - T0).count();
    
    // Calculate statistics
    double avg_bpm = 0.0;
    if (final_peaks.size() > 1) {
        vector<double> rr(final_peaks.size() - 1);
        for (size_t i = 1; i < final_peaks.size(); i++) {
            rr[i-1] = (final_peaks[i] - final_peaks[i-1]) / double(fs);
        }
        double mean_rr = accumulate(rr.begin(), rr.end(), 0.0) / rr.size();
        if (mean_rr > 0.0) avg_bpm = 60.0 / mean_rr;
    }
    
    // Output results
    cout << "\n--- RESULTS ---\n";
    cout << "Total samples: " << N << "\n";
    cout << "Detected R-peaks: " << final_peaks.size() << "\n";
    cout << "Avg BPM: " << fixed << setprecision(2) << avg_bpm << "\n";
    cout << "Filtering time (bandpass): " << fixed << setprecision(2) 
         << filter_time_ms << " ms\n";
    cout << "Pre-processing time (total): " << fixed << setprecision(2) 
         << preproc_time_ms << " ms\n";
    cout << "Chunked detection time (parallel): " << fixed << setprecision(2) 
         << detect_time_ms << " ms\n";
    cout << "Total processing time: " << fixed << setprecision(2) 
         << total_time_ms << " ms\n";
    
    cout << "\nFirst 10 R-peaks: ";
    for (size_t i = 0; i < final_peaks.size() && i < 10; i++) {
        cout << final_peaks[i] << " ";
    }
    cout << "\n";
    
    // Show chunk processing times
    cout << "\nChunk processing times:\n";
    for (int i = 0; i < min(5, (int)chunk_results.size()); i++) {
        cout << "  Chunk " << i << ": " << fixed << setprecision(2) 
             << chunk_results[i].elapsed_ms << " ms, " 
             << chunk_results[i].peaks_global.size() << " peaks\n";
    }
    if (chunk_results.size() > 5) {
        cout << "  ... (" << chunk_results.size() - 5 << " more chunks)\n";
    }
    
    return 0;
}
