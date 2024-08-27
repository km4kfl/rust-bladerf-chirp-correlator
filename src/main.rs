use bladerf;
use std::vec::Vec;
use std::sync::Arc;
use std::ops::{Add, Mul};
use std::thread;
use num_complex::Complex;
use std::sync::mpsc::channel;
use std::time::{Instant, Duration};
use std::fs::{OpenOptions, File};
use byteorder::{LittleEndian, WriteBytesExt};
use std::io::{Cursor, Write, Seek, SeekFrom};
use rustfft;
use rand::prelude::*;

fn dft(x: &[Complex<f64>], out: &mut [Complex<f64>]) {
    let imm_step = -((2.0 * std::f64::consts::PI) / x.len() as f64);
    for k in 0..x.len() {
        let mut sum = Complex::<f64> { re: 0.0, im: 0.0 };
        for n in 0..x.len() {
            let theta = imm_step * (n as f64) * (k as f64);
            let imm = Complex::<f64> {
                re: 0.0,
                im: theta,
            }.exp();
            sum += x[n] * imm;
        }
        out[k] = sum / x.len() as f64;
    }
}

fn idft(x: &[Complex<f64>], out: &mut [Complex<f64>]) {
    let imm_step = (2.0 * std::f64::consts::PI) / x.len() as f64;
    for k in 0..x.len() {
        for n in 0..x.len() {
            let theta = imm_step * (n as f64) * (k as f64);
            let imm = Complex::<f64> {
                re: 0.0,
                im: theta,
            }.exp();
            out[n] += imm * x[k];            
        }
    }
}

fn linspace64(start: f64, end: f64, steps: u32) -> Vec<f64> {
    let mut out: Vec<f64> = vec!(0.0f64; steps as usize);
    let delta = end - start;
    
    let mut step = 0u32;
    for out_v in out.iter_mut() {
        let p = (step as f64) / (steps as f64 - 1.0);
        let cur = start + p * delta;
        *out_v = cur;
        step += 1;
    }

    out
}

fn linspace32(start: f32, end: f32, steps: u32) -> Vec<f32> {
    let mut out: Vec<f32> = vec!(0.0f32; steps as usize);
    let delta = end - start;
    
    let mut step = 0u32;
    for out_v in out.iter_mut() {
        let p = (step as f32) / (steps as f32 - 1.0);
        let cur = start + p * delta;
        *out_v = cur;
        step += 1;
    }

    out
}

fn roll_slice_left(a: &mut [Complex<f64>]) {
    a[a.len() - 1] = a[0];
    for x in 0..a.len() - 1 {
        a[x] = a[x+1];
    }
}

fn multiply_slice_offset_wrapping_sum<T: num_traits::Float + Default + std::ops::AddAssign + std::ops::SubAssign + std::ops::MulAssign + std::ops::DivAssign + std::ops::RemAssign>(
    a: &[Complex<T>],
    b: &[Complex<T>],
    offset: usize) -> Complex<T> {
    let mut sum = Complex::<T> { re: T::default(), im: T::default() };
    for y in 0..a.len() {
        sum += a[y] * b[(y + offset) % b.len()];
    }

    sum
}

fn multiply_slice(a: &[Complex<f64>], b: &[Complex<f64>], c: &mut [Complex<f64>]) {
    for ((va, vb), vc) in a.iter().zip(b.iter()).zip(c.iter_mut()) {
        *vc = va * vb;
    }
}

fn conjugate_slice64(a: &mut [Complex<f64>]) {
    for va in a.iter_mut() {
        *va = va.conj();
    }
}

fn conjugate_slice32(a: &mut [Complex<f32>]) {
    for va in a.iter_mut() {
        *va = va.conj();
    }
}

/// Return the maximum magnitude along the slice of complex values.
fn max_iq_slice<T: Default + num_traits::Float>(a: &[Complex<T>]) -> T {
    let mut mag_max = T::default();
    for v in a.iter() {
        let mag = T::sqrt(v.norm_sqr());
        if mag > mag_max {
            mag_max = mag;
        }
    }

    mag_max
}

fn sum_iq_slice<T: Default + num_traits::Float + std::ops::AddAssign>(a: &[Complex<T>]) -> Complex<T> {
    let mut out = Complex::<T> {
        re: T::default(),
        im: T::default(),
    };

    for va in a.iter() {
        out.re += va.re;
        out.im += va.im;
    }

    out
}

/// Convert a slice of Complex<i16> to Complex<f64>. 
///
/// This does not scale the values.
fn convert_iqi16_to_iqf64(a: &[Complex<i16>], out: &mut [Complex<f64>]) {
    for (av, out_v) in a.iter().zip(out.iter_mut()) {
        (*out_v).re = av.re as f64;
        (*out_v).im = av.im as f64;
    }
}

fn convert_iqi16_to_iqf32(a: &[Complex<i16>], out: &mut [Complex<f32>]) {
    for (av, out_v) in a.iter().zip(out.iter_mut()) {
        (*out_v).re = av.re as f32;
        (*out_v).im = av.im as f32;
    }
}

/// Add the slice of complex values to the scalar and return a new vector with the output.
fn add_sf64_scalar(a: &[f64], b: f64) -> Vec<f64> {
    let mut out: Vec<f64> = Vec::with_capacity(a.len());
    for v in a.iter() {
        out.push(v + b);
    }

    out
}

/// Multiply the slice of complex values by the scalar.
fn mul_sf64_scalar(a: &[f64], b: f64) -> Vec<f64> {
    let mut out: Vec<f64> = Vec::with_capacity(a.len());
    for v in a.iter() {
        out.push(v * b);
    }

    out
}

fn mul_sf32_scalar(a: &[f32], b: f32) -> Vec<f32> {
    let mut out: Vec<f32> = Vec::with_capacity(a.len());
    for v in a.iter() {
        out.push(v * b);
    }

    out
}

/// Divide the slice of complex values by the scalar.
fn div_iqf64_scalar_inplace(a: &mut [Complex<f64>], b: f64) {
    for v in a.iter_mut() {
        *v /= b;
    }
}

fn div_iqf32_scalar_inplace(a: &mut [Complex<f32>], b: f32) {
    for v in a.iter_mut() {
        *v /= b;
    }
}

/// Return the frequency bin centers for rustfft's FFT.
fn fftfreq(sz: usize) -> Vec<f64> {
    let mut out: Vec<f64> = Vec::with_capacity(sz);
    let spacing = 1.0 / sz as f64;
    
    out.push(0.0);

    if sz % 2 != 0 {
        let g = ((sz - 1) / 2) as u32;
        let a = mul_sf64_scalar(&linspace64(1.0, g as f64, g), spacing);
        let b = mul_sf64_scalar(&linspace64(-(g as f64), -1.0, g), spacing);
        for v in b.iter() {
            out.push(*v);
        }        
        for v in a.iter() {
            out.push(*v);
        }
    } else {
        let g = (sz / 2) as u32;
        let a = mul_sf64_scalar(&linspace64(1.0, g as f64 - 1.0, g - 1), spacing);
        let b = mul_sf64_scalar(&linspace64(-(g as f64), -1.0, g), spacing);
        for v in b.iter() {
            out.push(*v);
        }        
        for v in a.iter() {
            out.push(*v);
        }
    }

    out
}

/// Shifts the signal components by a phase that is a function of distance and frequency.
///
/// As a signal travels it's individual frequency components change as a function of distance
/// and this function tries to estimate that change based on frequency and distance.
fn phase_shift_signal_by_distance(
    sps: f64,
    center_freq: f64,
    signal_in: &[Complex<f32>],
    dist: f64,
    wave_velocity: f64) -> Vec<Complex<f32>> {
    let mut planner = rustfft::FftPlanner::new();
    let fft_forward = planner.plan_fft(signal_in.len(), rustfft::FftDirection::Forward);
    let fft_backward = planner.plan_fft(signal_in.len(), rustfft::FftDirection::Inverse);
    
    let mut signal = Vec::with_capacity(signal_in.len());
    signal.extend_from_slice(signal_in);

    fft_forward.process(&mut signal);

    let freqs = fftfreq(signal.len());
    
    for i in 0..freqs.len() {
        let phase_distance = dist / (wave_velocity / (freqs[i] * sps * 0.5 + center_freq)) * std::f64::consts::PI * 2.0;
        signal[i] *= (Complex::<f32> {
            re: 0.0,
            im: (phase_distance % (std::f64::consts::PI * 2.0)) as f32,
        }).exp();
    }

    fft_backward.process(&mut signal);

    signal
}

/// Performs an FFT correlation. Equivalent to scipy.signal.correlate(a, b, mode='same').
///
/// Wraps the `rustfft` planner and transforms to reduce run-time for a constant size input.
struct FftCorrelate32 {
    fft_size: usize,
    planner: rustfft::FftPlanner<f32>,
    forward: Arc<dyn rustfft::Fft<f32>>,
    inverse: Arc<dyn rustfft::Fft<f32>>,
}

impl FftCorrelate32 {
    fn new(a_size: usize, b_size: usize) -> FftCorrelate32 {
        let fft_size = a_size + b_size+ 1;
        let mut planner = rustfft::FftPlanner::new();
        let forward = planner.plan_fft(fft_size, rustfft::FftDirection::Forward);
        let inverse = planner.plan_fft(fft_size, rustfft::FftDirection::Inverse);
        FftCorrelate32 {
            fft_size: fft_size,
            planner: planner,
            forward: forward,
            inverse: inverse,
        }
    }

    fn correlate(
        &self, 
        a: &[Complex<f32>],
        b: &[Complex<f32>]) -> Vec<Complex<f32>>
    {
        let mut ap = vec!(
            Complex::<f32> {
                re: 0.0, im: 0.0 
            }; self.fft_size
        );
        
        let mut bp = vec!(
            Complex::<f32> {
                re: 0.0, im: 0.0
            }; self.fft_size
        );

        for (avv, av) in ap.iter_mut().zip(a.iter()) {
            *avv = *av;
        }

        for (bvv, bv) in bp.iter_mut().zip(b.iter()) {
            *bvv = *bv;
        }

        self.forward.process(&mut ap);
        self.forward.process(&mut bp);

        for (av, bv) in ap.iter_mut().zip(bp.iter()) {
            *av = *av * bv.conj();
        }

        self.inverse.process(&mut ap);
        
        let adj = (b.len() - 1) * 2;
        let valid = a.len() - b.len() + 1;

        let mut out: Vec<Complex<f32>> = Vec::with_capacity(
            valid
        );

        for x in 0..valid {
            out.push(
                ap[(adj + x) % ap.len()] / ap.len() as f32
            );
        }
        
        out
    }
}

struct FftCorrelate64 {
    fft_size: usize,
    planner: rustfft::FftPlanner<f64>,
    forward: Arc<dyn rustfft::Fft<f64>>,
    inverse: Arc<dyn rustfft::Fft<f64>>,
}

impl FftCorrelate64 {
    fn new(a_size: usize, b_size: usize) -> FftCorrelate64 {
        let fft_size = a_size + b_size+ 1;
        let mut planner = rustfft::FftPlanner::new();
        let forward = planner.plan_fft(fft_size, rustfft::FftDirection::Forward);
        let inverse = planner.plan_fft(fft_size, rustfft::FftDirection::Inverse);
        FftCorrelate64 {
            fft_size: fft_size,
            planner: planner,
            forward: forward,
            inverse: inverse,
        }
    }

    fn correlate(
        &self, 
        a: &[Complex<f64>],
        b: &[Complex<f64>]) -> Vec<Complex<f64>>
    {
        let mut ap = vec!(
            Complex::<f64> {
                re: 0.0, im: 0.0 
            }; self.fft_size
        );
        
        let mut bp = vec!(
            Complex::<f64> {
                re: 0.0, im: 0.0
            }; self.fft_size
        );

        for (avv, av) in ap.iter_mut().zip(a.iter()) {
            *avv = *av;
        }

        for (bvv, bv) in bp.iter_mut().zip(b.iter()) {
            *bvv = *bv;
        }

        self.forward.process(&mut ap);
        self.forward.process(&mut bp);

        for (av, bv) in ap.iter_mut().zip(bp.iter()) {
            *av = *av * bv.conj();
        }

        self.inverse.process(&mut ap);
        
        let adj = (b.len() - 1) * 2;
        let valid = a.len() - b.len() + 1;

        let mut out: Vec<Complex<f64>> = Vec::with_capacity(
            valid
        );

        for x in 0..valid {
            out.push(
                ap[(adj + x) % ap.len()] / ap.len() as f64
            );
        }
        
        out
    }
}


fn main() {
    println!("opening bladerf..");
    let dev = bladerf::open(None).expect("card should have been opened");
    println!(
        "fpga:{:?} firmware:{:?}",
        dev.fpga_version().expect("should have returned fpga version"),
        dev.fw_version().expect("should have returned firmware version")
    );

    // The samples per second.
    //let sps = 520834u32;
    let sps = 4_000_000u32;
    // The size of the chirp in samples.
    let samps: usize = 1024 * 16;
    // The chirp start frequency.
    let freq_start = 200e3f32;
    // The chirp end frequency.
    let freq_end = 240e3f32;
    // The number of scan points. This is the number of points at which
    // the chirp correlation is evaluated. The more the further the distance
    // and greater the time.
    let sample_distance = 400usize;
    // The speed of light or some fraction of it if you desire.
    let wave_velocity = 299792458.0f64;

    dev.set_frequency(bladerf::bladerf_module::RX0, 4_000_000_000).expect(
        "should have set the RX frequency"
    );
    dev.set_frequency(bladerf::bladerf_module::TX0, 4_000_000_000).expect(
        "should have set the TX frequency"
    );

    dev.set_gain(bladerf::bladerf_module::TX0, 20).expect("TX0 gain set");
    dev.set_gain(bladerf::bladerf_module::TX1, 0).expect("TX1 gain set");
    dev.set_gain(bladerf::bladerf_module::RX0, 60).expect("RX0 gain set");
    dev.set_gain(bladerf::bladerf_module::RX1, 0).expect("RX1 gain set");

    dev.set_bandwidth(bladerf::bladerf_module::TX0, sps).expect("TX0 bw set");
    dev.set_bandwidth(bladerf::bladerf_module::TX1, sps).expect("TX1 bw set");
    dev.set_bandwidth(bladerf::bladerf_module::RX0, sps).expect("RX0 bw set");
    dev.set_bandwidth(bladerf::bladerf_module::RX1, sps).expect("RX1 bw set");

    dev.set_sample_rate(bladerf::bladerf_module::RX0, 520834).expect(
        "RX0/RX1 sampling rate set"
    );

    dev.set_sample_rate(bladerf::bladerf_module::TX0, 520834).expect(
        "TX0/TX1 sampling rate set"
    );

    let num_buffers = 16u32;
    let buffer_size = 4096u32;
    let num_transfers = 8u32;
    let stream_timeout = 20000u32;

    let buffer_samps = num_buffers * buffer_size / 4;

    dev.sync_config(
        bladerf::bladerf_channel_layout::RX_X1,
        bladerf::bladerf_format::SC16_Q11_META,
        num_buffers, buffer_size,
        Some(num_transfers),
        stream_timeout
    ).expect("sync_config for rx");

    dev.sync_config(
        bladerf::bladerf_channel_layout::TX_X1,
        bladerf::bladerf_format::SC16_Q11_META,
        num_buffers, buffer_size,
        Some(num_transfers),
        stream_timeout
    ).expect("sync_config for tx");

    dev.enable_module(bladerf::bladerf_module::RX0, true).expect(
        "rx0 module enable"
    );

    //thread::sleep(Duration::from_micros(200));

    dev.enable_module(bladerf::bladerf_module::TX0, true).expect(
        "tx0 module enable"
    );

    let mut signal: Vec<Complex<f32>> = vec!(Complex::<f32> { re: 0.0f32, im: 0.0f32 }; samps);
    let mut tx_data: Vec<Complex<i16>> = vec!(Complex::<i16> { re: 0i16, im: 0i16 }; samps);
    let theta_step_start = freq_start * std::f32::consts::PI * 2.0f32 / (sps as f32);
    let theta_step_end = freq_end * std::f32::consts::PI * 2.0f32 / (sps as f32);
    let t_space = linspace32(theta_step_start, theta_step_end, samps as u32);

    {
        let mut theta = 0.0f32;
        for x in 0..samps {
            signal[x].re = f32::cos(theta);
            signal[x].im = f32::sin(theta);
            tx_data[x].re = (signal[x].re * 2000.0) as i16;
            tx_data[x].im = (signal[x].im * 2000.0) as i16;
            theta += t_space[x];
            theta = theta % (std::f32::consts::PI * 2.0);
        }
    }
    
    let dev_arc = Arc::new(dev);

    let dev_arc_tx = dev_arc.clone();

    let (rx_tx, tx_rx) = channel::<bool>();
    let (tx_tx, rx_rx) = channel::<u64>();

    let tx_handler = thread::spawn(move || {
        let initial_timestamp = dev_arc_tx.get_timestamp(bladerf::bladerf_module::TX0) + 
                                (sps as f64 * 0.100f64) as u64;

        tx_tx.send(initial_timestamp).expect("tx sending initial timestamp value");

        let mut meta = bladerf::Struct_bladerf_metadata {
            timestamp: initial_timestamp,
            flags: bladerf::bladerf_meta_tx::FLAG_TX_BURST_START as u32,
            status: 0,
            actual_count: 0,
            reserved: [0u8; 32],
        };

        loop {
            dev_arc_tx.sync_tx_meta(&tx_data, &mut meta, 20000).expect("sync_tx");

            //println!("tx:timestamp:{:} actual_count:{:}", meta.timestamp, meta.actual_count);

            match tx_rx.recv_timeout(Duration::from_millis(0)) {
                Ok(_) => break,
                Err(_) => (),
            };

            meta.flags = 0;
        }
    });

    //tx_handler.join().expect("joined tx thread");

    let mut rx_data = vec!(
        Complex::<i16> { re: 0i16, im: 0i16 };
        samps * 2
    );

    let mut rx_signal = vec!(
        Complex::<f32> {
            re: 0.0,
            im: 0.0,
        }; rx_data.len()
    );

    let mut rx_trash = vec!(
        Complex::<i16> {
            re: 0i16,
            im: 0i16,
        }; buffer_samps as usize
    );

    let st = Instant::now();

    let mut fout = File::create("out.bin").expect("opening output file");
    //let mut fout = OpenOptions::new().write(true).open("out.bin").expect("opening output file");

    let mut fout_buffer = Cursor::new(vec!(0u8; 4));

    let correlate = FftCorrelate32::new(rx_data.len(), samps);

    //let small_correlate = FftCorrelate::new(samps, samps);

    let initial_timestamp = rx_rx.recv().expect("sync tx reply with initial timestamp");

    let mut avg_diff = 0f64;
    let mut avg_diff_cnt = 0f64;

    let mut cycle = 0usize;

    let avg_buf_lines = 32;
    let mut avg_buf = vec!(0f32; sample_distance * avg_buf_lines);

    let mut rng = rand::thread_rng();

    let freq_start = 800e6f64;
    let freq_end = 6000e6f64;

    loop {
        let freq = 3_611_697_000.0f64; //rng.gen::<f64>() * (freq_end - freq_start) + freq_start;
        
        dev_arc.set_frequency(bladerf::bladerf_module::RX0, freq as u64).expect(
            "should have set the RX frequency"
        );
        dev_arc.set_frequency(bladerf::bladerf_module::TX0, freq as u64).expect(
            "should have set the TX frequency"
        );

        println!("set frequency {:}", freq);

        let mut signal_slots: Vec<Vec<Complex<f32>>> = Vec::new();
        let distance_per_sample = wave_velocity / sps as f64;
    
        for x in 0..sample_distance {
            //println!("building cor {:}", x);
            let distance = distance_per_sample * x as f64;
            let mut signal_shifted = phase_shift_signal_by_distance(
                sps as f64,
                freq as f64,
                &signal,
                distance,
                wave_velocity
            );
            //let mut signal_shifted = signal.clone();
            conjugate_slice32(&mut signal_shifted);
            signal_slots.push(signal_shifted);
        }

        loop {
            let mut rx_datas: Vec<(usize, Vec<Complex<i16>>)> = Vec::new();

            for cycle in 0..avg_buf_lines {
                // Clear the buffer.
                let mut meta;
                
                meta = bladerf::Struct_bladerf_metadata::default();
                meta.flags = bladerf::bladerf_meta_rx::FLAG_RX_NOW as u32;
                dev_arc.sync_rx_meta(&mut rx_trash, &mut meta, 20000).expect("rx sync call [trash]");

                meta = bladerf::Struct_bladerf_metadata::default();
                meta.flags = bladerf::bladerf_meta_rx::FLAG_RX_NOW as u32;
                dev_arc.sync_rx_meta(&mut rx_data, &mut meta, 20000).expect("rx sync call [data]");

                // Calculate the offset of the first chirp in the buffer. I'm assuming the RX and TX use the
                // same sample counter in the FPGA.
                let best_ndx = samps - ((meta.timestamp - initial_timestamp) % samps as u64) as usize;

                rx_datas.push((best_ndx, rx_data.clone()));
            }

            println!("rx_datas.len():{:}", rx_datas.len());

            for cycle in 0..avg_buf_lines {
                //println!("rx:timestamp:{:} actual_count:{:} best_ndx:{:}", meta.timestamp, meta.actual_count, best_ndx);
                let best_ndx = rx_datas[cycle].0;

                // Convert the 16-bit signed complex to 64-bit floating point complex.
                convert_iqi16_to_iqf32(&rx_datas[cycle].1, &mut rx_signal);
                // Scale it down so the correlation sums don't get larger than needed.
                div_iqf32_scalar_inplace(&mut rx_signal, 2896.309);

                {
                    let max = max_iq_slice(&rx_signal);
                    let mut avg = 0.0f32;
                    for v in rx_signal.iter() {
                        avg += f32::sqrt(v.norm_sqr());
                    }
                    avg /= rx_data.len() as f32;
                    //println!("rx_data peak:{:} avg:{:}", max, avg);
                }        

                // This was the OLD way. Now, we use the timestamp to synchronize the RX and TX.
                
                // This does an FFT based correlation. Python: scipy.signal.correlate(rx_signal, signal, mode='same').
                let initial_cor = correlate.correlate(&rx_signal, &signal);

                // The highest peak *should* be the initial TX chirp signal and everything
                // after should be the reflections. Find this initial index.
                let mut n_best_mag = 0f32;
                let mut n_best_ndx = 0usize;
                for i in 0..initial_cor.len() {
                    let mag = f32::sqrt(initial_cor[i].norm_sqr());
                    if mag > n_best_mag {
                        n_best_mag = mag;
                        n_best_ndx = i;
                    }
                }

                {
                    let diff = best_ndx as isize - n_best_ndx as isize;
                    avg_diff += diff as f64;
                    avg_diff_cnt += 1.0;
                    //println!("processing best_ndx:{:} n_best_ndx:{:} diff:{:}", best_ndx, n_best_ndx, avg_diff / avg_diff_cnt);
                }

                for i in 0..sample_distance {
                    let ss = &signal_slots[i];
                    
                    // This is slower than the direct method.
                    //let mag = f64::sqrt(small_correlate.correlate(&rx_signal[best_ndx+i..best_ndx+i+samps], &ss)[0].norm_sqr());

                    let sample = multiply_slice_offset_wrapping_sum::<f32>(
                        &ss, &rx_signal, best_ndx + i
                    );
                    let mag = f32::sqrt(sample.norm_sqr());
                    let theta = sample.arg();

                    if i == 0 {
                        println!("mag[0]:{:}", mag);
                    }

                    avg_buf[sample_distance * cycle + i] = mag;
                }
            }

            println!("writing to file");
            fout_buffer.seek(SeekFrom::Start(0)).expect("seeking into buffer");
            fout_buffer.write_f32::<LittleEndian>((freq / 1e3) as f32).expect("encoding f64 into bytes");
            fout.write(fout_buffer.get_ref()).expect("writing magnitude to file");

            for i in 0..sample_distance {
                let mut avg = 0f32;
                for y in 0..avg_buf_lines {
                    avg += avg_buf[sample_distance * y + i];
                }
                avg /= avg_buf_lines as f32;
                // Write the value to the file.
                fout_buffer.seek(SeekFrom::Start(0)).expect("seeking into buffer");
                fout_buffer.write_f32::<LittleEndian>(avg).expect("encoding f64 into bytes");
                fout.write(fout_buffer.get_ref()).expect("writing magnitude to file");
            }
        }
    }

    rx_tx.send(true).expect("tried sending tx thread shutdown command");

    println!("joining tx thread");
    tx_handler.join().expect("joining tx thread");
}
