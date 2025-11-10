#include <bits/stdc++.h>
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

// ---------------- Bandpass Filter (5-18 Hz) ----------------
vector<double> bandpass_filter(const vector<double>& x, int fs) {
    int N = (int)x.size();
    vector<double> y(N, 0.0);
    
    // Simple IIR bandpass approximation (5-18 Hz)
    // Low-pass component
    double alpha_lp = 0.15; // ~18 Hz cutoff
    for (int i = 1; i < N; i++) {
        y[i] = alpha_lp * x[i] + (1 - alpha_lp) * y[i-1];
    }
    
    // High-pass component (subtract low-pass with lower cutoff)
    vector<double> lp_low(N, 0.0);
    double alpha_hp = 0.05; // ~5 Hz cutoff
    for (int i = 1; i < N; i++) {
        lp_low[i] = alpha_hp * y[i] + (1 - alpha_hp) * lp_low[i-1];
    }
    
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

// ---------------- Derivative ----------------
vector<double> derivative(const vector<double>& x, int fs) {
    int N = (int)x.size();
    vector<double> y(N, 0.0);
    double T = 1.0 / fs;
    
    for (int n = 2; n < N - 2; n++) {
        y[n] = (1.0 / (8.0 * T)) * (-x[n-2] - 2*x[n-1] + 2*x[n+1] + x[n+2]);
    }
    return y;
}

// ---------------- Savitzky-Golay Filter (3rd order, window=11) ----------------
vector<double> savitzky_golay_filter(const vector<double>& x) {
    int N = (int)x.size();
    vector<double> y(N, 0.0);
    
    // Coefficients for 3rd order, 11-point SG filter
    vector<double> coeff = {-36, 9, 44, 69, 84, 89, 84, 69, 44, 9, -36};
    double norm = 429.0;
    
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

// ---------------- Squaring ----------------
vector<double> squaring(const vector<double>& x) {
    int N = (int)x.size();
    vector<double> y(N);
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

// ---------------- Improved Pan-Tompkins QRS Detection ----------------
vector<int> detect_qrs_improved(const vector<double>& integrated, 
                                const vector<double>& filtered, 
                                int fs) {
    int N = (int)integrated.size();
    vector<int> peaks;
    
    // Initialize thresholds based on first 2 seconds
    int init_samples = min(2 * fs, N);
    double max_h = *max_element(integrated.begin(), integrated.begin() + init_samples);
    double mean_h = accumulate(integrated.begin(), integrated.begin() + init_samples, 0.0) / init_samples;
    
    double THR1 = max_h / 3.0;
    double THR2 = 0.5 * mean_h;
    double THR3 = 0.5 * THR2;
    double ESP = THR1;
    double ENP = THR2;
    
    int min_interval = int(0.231 * fs); // 231 ms = 260 bpm max
    int last_peak = -min_interval;
    
    deque<int> rr_intervals;
    
    // First pass detection
    for (int i = 1; i < N - 1; i++) {
        // Local maximum
        if (integrated[i] > integrated[i-1] && integrated[i] >= integrated[i+1]) {
            
            if (integrated[i] > THR1 && (i - last_peak) > min_interval) {
                // Check slope if interval too short
                bool valid = true;
                if (!peaks.empty() && (i - last_peak) < int(0.36 * fs)) {
                    int window = int(0.07 * fs);
                    int prev_peak = peaks.back();
                    
                    double curr_slope = 0.0, prev_slope = 0.0;
                    for (int j = max(0, i - window); j < min(N, i + window); j++) {
                        if (j > 0) curr_slope = max(curr_slope, abs(filtered[j] - filtered[j-1]));
                    }
                    for (int j = max(0, prev_peak - window); j < min(N, prev_peak + window); j++) {
                        if (j > 0) prev_slope = max(prev_slope, abs(filtered[j] - filtered[j-1]));
                    }
                    
                    if (curr_slope < 0.6 * prev_slope) {
                        valid = false; // Likely T-wave
                    }
                }
                
                if (valid) {
                    peaks.push_back(i);
                    
                    // Update RR intervals
                    if (!rr_intervals.empty()) {
                        rr_intervals.push_back(i - last_peak);
                        if (rr_intervals.size() > 8) rr_intervals.pop_front();
                    } else {
                        rr_intervals.push_back(i - last_peak);
                    }
                    
                    last_peak = i;
                    
                    // Update thresholds (adaptive - faster response)
                    ESP = 0.75 * integrated[i] + 0.25 * ESP;
                    THR1 = ENP + 0.25 * (ESP - ENP);
                    THR2 = 0.4 * THR1;
                }
            }
        }
    }
    
    // Search-back for missed peaks
    vector<int> additional_peaks;
    double mean_rr = 0.0;
    if (!rr_intervals.empty()) {
        mean_rr = accumulate(rr_intervals.begin(), rr_intervals.end(), 0.0) / rr_intervals.size();
    }
    
    for (size_t k = 0; k < peaks.size() - 1; k++) {
        int gap = peaks[k+1] - peaks[k];
        
        // If gap > 1s or > 166% of mean RR
        if (gap > fs || (mean_rr > 0 && gap > 1.66 * mean_rr)) {
            int search_start = peaks[k] + int(0.36 * fs);
            int search_end = peaks[k+1];
            
            // Find peak in this region
            for (int i = search_start; i < search_end; i++) {
                if (i > 0 && i < N-1 && 
                    integrated[i] > integrated[i-1] && 
                    integrated[i] >= integrated[i+1] &&
                    integrated[i] > THR3) {
                    
                    additional_peaks.push_back(i);
                    
                    // Update ESP/ENP
                    ESP = 0.75 * integrated[i] + 0.25 * ESP;
                    ENP = 0.75 * integrated[i] + 0.25 * ENP;
                    break;
                }
            }
        }
    }
    
    // Merge additional peaks
    peaks.insert(peaks.end(), additional_peaks.begin(), additional_peaks.end());
    sort(peaks.begin(), peaks.end());
    
    // Fallback for very long gaps (>1.4s)
    vector<int> fallback_peaks;
    for (size_t k = 0; k < peaks.size() - 1; k++) {
        int gap = peaks[k+1] - peaks[k];
        if (gap > int(1.4 * fs)) {
            int search_start = peaks[k] + int(0.36 * fs);
            int search_end = peaks[k+1];
            
            for (int i = search_start; i < search_end; i++) {
                if (i > 0 && i < N-1 && 
                    integrated[i] > integrated[i-1] && 
                    integrated[i] >= integrated[i+1] &&
                    integrated[i] > 0.2 * THR2) {
                    fallback_peaks.push_back(i);
                    break;
                }
            }
        }
    }
    
    peaks.insert(peaks.end(), fallback_peaks.begin(), fallback_peaks.end());
    sort(peaks.begin(), peaks.end());
    peaks.erase(unique(peaks.begin(), peaks.end()), peaks.end());
    
    return peaks;
}

// ---------------- Main ----------------
int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <path_to_ecg_csv>\n";
        return 1;
    }
    
    string filepath = argv[1];
    int fs = 360; // Sampling frequency
    
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
    auto filtered = bandpass_filter(ecg, fs);
    
    // Step 2: Derivative
    cout << "Computing derivative...\n";
    auto diff = derivative(filtered, fs);
    
    // Step 3: Savitzky-Golay smoothing (after derivative)
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
    
    // Step 7: Adaptive QRS detection
    cout << "Detecting QRS complexes with improved Pan-Tompkins...\n";
    auto peaks = detect_qrs_improved(integrated, filtered, fs);
    
    auto T1 = hrc_t::now();
    double total_time_ms = chrono::duration<double, milli>(T1 - T0).count();
    
    // Calculate statistics
    double avg_bpm = 0.0;
    if (peaks.size() > 1) {
        vector<double> rr(peaks.size() - 1);
        for (size_t i = 1; i < peaks.size(); i++) {
            rr[i-1] = (peaks[i] - peaks[i-1]) / double(fs);
        }
        double mean_rr = accumulate(rr.begin(), rr.end(), 0.0) / rr.size();
        if (mean_rr > 0.0) avg_bpm = 60.0 / mean_rr;
    }
    
    // Output results
    cout << "\n--- RESULTS ---\n";
    cout << "Total samples: " << N << "\n";
    cout << "Detected R-peaks: " << peaks.size() << "\n";
    cout << "Avg BPM: " << fixed << setprecision(2) << avg_bpm << "\n";
    cout << "Total processing time: " << fixed << setprecision(2) 
         << total_time_ms << " ms\n";
    cout << "Processing time per sample: " << fixed << setprecision(6) 
         << (total_time_ms / N) << " ms\n";
    
    cout << "\nFirst 10 R-peaks: ";
    for (size_t i = 0; i < peaks.size() && i < 10; i++) {
        cout << peaks[i] << " ";
    }
    cout << "\n";
    
    return 0;
}
