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
use sigrs::*;

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
        for v in a.iter() {
            out.push(*v);
        }        
        for v in b.iter() {
            out.push(*v);
        }        
    } else {
        let g = (sz / 2) as u32;
        let a = mul_sf64_scalar(&linspace64(1.0, g as f64 - 1.0, g - 1), spacing);
        let b = mul_sf64_scalar(&linspace64(-(g as f64), -1.0, g), spacing);
        for v in a.iter() {
            out.push(*v);
        }        
        for v in b.iter() {
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
    signal_in: &[Complex<f64>],
    dist: f64,
    wave_velocity: f64,
    fft_forward: &Arc<dyn rustfft::Fft<f64>>,
    fft_backward: &Arc<dyn rustfft::Fft<f64>>) 
    -> Vec<Complex<f64>> {    
    let mut signal = Vec::with_capacity(signal_in.len());
    signal.extend_from_slice(signal_in);

    fft_forward.process(&mut signal);

    let freqs = fftfreq(signal.len());
    
    for i in 0..freqs.len() {
        let phase_distance = dist / (wave_velocity / (freqs[i] * sps * 0.5 + center_freq)) * std::f64::consts::PI * 2.0;
        signal[i] *= (Complex::<f64> {
            re: 0.0,
            im: (phase_distance % (std::f64::consts::PI * 2.0)),
        }).exp();
    }

    fft_backward.process(&mut signal);

    signal
}

/// Returns an exponential space over the start and end specified.
fn expspace64(start: f64, end: f64, count: usize, exp: f64) -> Vec<f64> {
    let mut base = linspace64(0f64, count as f64, count as u32);

    for x in 0..base.len() {
        base[x] = f64::powf(base[x], exp);
    }

    let base_max = base[base.len() - 1];

    for x in 0..base.len() {
        base[x] = (base[x] / base_max) * (base_max - base[x]) + base[0];
    }

    return base;
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
    let sps = 520834u32;
    //let sps = 40_000_000;
    // The size of the chirp in samples.
    let samps: usize = 1024 * 32;
    // The chirp start frequency.
    let freq_start = -200e3f64;
    // The chirp end frequency.
    let freq_end = 200e3f64;
    let bw = sps;
    // The number of scan points. This is the number of points at which
    // the chirp correlation is evaluated. The more the further the distance
    // and greater the time.
    let doppler_shift_min = -5e3f64;
    let doppler_shift_max = 5e3f64;
    let doppler_shift_steps = 40usize;
    let sample_distance = 3000usize;
    // The speed of light or some fraction of it if you desire.
    let wave_velocity = 299792458.0f64;
    //let freq = 1_830_124_000.0f64;
    let freq = 3_000_000_000.0f64;
    // It should be better to perform averaging over all the data but
    // if it is producing too much output you can slow it down by averaging
    // some of it together.
    let avg_buf_lines = 1;
    
    dev.set_frequency(bladerf::bladerf_module::RX0, freq as u64).expect(
        "should have set the RX frequency"
    );
    dev.set_frequency(bladerf::bladerf_module::TX0, freq as u64).expect(
        "should have set the TX frequency"
    );

    println!("set frequency {:}", freq);

    dev.set_gain(bladerf::bladerf_module::TX0, 60).expect("TX0 gain set");
    dev.set_gain(bladerf::bladerf_module::TX1, 0).expect("TX1 gain set");
    dev.set_gain(bladerf::bladerf_module::RX0, 60).expect("RX0 gain set");
    dev.set_gain(bladerf::bladerf_module::RX1, 0).expect("RX1 gain set");

    dev.set_bandwidth(bladerf::bladerf_module::TX0, bw).expect("TX0 bw set");
    dev.set_bandwidth(bladerf::bladerf_module::TX1, bw).expect("TX1 bw set");
    dev.set_bandwidth(bladerf::bladerf_module::RX0, bw).expect("RX0 bw set");
    dev.set_bandwidth(bladerf::bladerf_module::RX1, bw).expect("RX1 bw set");

    dev.set_sample_rate(bladerf::bladerf_module::RX0, sps).expect(
        "RX0/RX1 sampling rate set"
    );

    dev.set_sample_rate(bladerf::bladerf_module::TX0, sps).expect(
        "TX0/TX1 sampling rate set"
    );

    println!("bias_tee:{:}", dev.get_bias_tee(bladerf::bladerf_module::TX0).unwrap());

    dev.set_bias_tee(bladerf::bladerf_module::TX0, true).expect(
        "set bias tee on TX0"
    );

    println!("bias_tee:{:}", dev.get_bias_tee(bladerf::bladerf_module::TX0).unwrap());

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

    let mut signal: Vec<Complex<f64>> = vec!(Complex::<f64> { re: 0.0, im: 0.0 }; samps);
    let mut tx_data: Vec<Complex<i16>> = vec!(Complex::<i16> { re: 0i16, im: 0i16 }; samps);
    let theta_step_start = freq_start * std::f64::consts::PI * 2.0f64 / (sps as f64);
    let theta_step_end = freq_end * std::f64::consts::PI * 2.0f64 / (sps as f64);
    //let t_space = linspace64(theta_step_start, theta_step_end, samps as u32);
    let t_space = expspace64(theta_step_start, theta_step_end, samps, 1.2);

    {
        let mut theta = 0.0f64;
        for x in 0..samps {
            signal[x].re = f64::cos(theta);
            signal[x].im = f64::sin(theta);
            tx_data[x].re = (signal[x].re * 2000.0) as i16;
            tx_data[x].im = (signal[x].im * 2000.0) as i16;
            theta += t_space[x];
            theta = theta % (std::f64::consts::PI * 2.0);
        }
    }

    let mut shifted_copies: Vec<Vec<Complex<f64>>> = Vec::with_capacity(doppler_shift_steps);
    
    let doppler_shift_amounts = linspace64(doppler_shift_min, doppler_shift_max, doppler_shift_steps as u32);

    for x in 0..doppler_shift_steps {
        let doppler_shift_amount = doppler_shift_amounts[x];
        let mut theta = 0.0f64;
        let theta_step = doppler_shift_amount * std::f64::consts::PI * 2.0 / sps as f64;
        let mut out_signal: Vec<Complex<f64>> = Vec::with_capacity(samps);
        for x in 0..samps {
            let sample = Complex::<f64> {
                re: 0.0,
                im: theta,
            }.exp();
            out_signal.push(signal[x] * sample);
            theta += theta_step;
            theta = theta % (std::f64::consts::PI * 2.0);
        }
        shifted_copies.push(out_signal);
    }

    let dev_arc = Arc::new(dev);

    let dev_arc_tx = dev_arc.clone();

    let (rx_tx, tx_rx) = channel::<bool>();
    let (tx_tx, rx_rx) = channel::<u64>();

    let tx_handler = thread::spawn(move || {
        let mut ts = dev_arc_tx.get_timestamp(bladerf::bladerf_module::TX0);

        loop {
            ts += (sps as f64 * 2.0f64) as u64;

            tx_tx.send(ts).expect("tx sending initial timestamp value");            
            
            let mut meta = bladerf::Struct_bladerf_metadata {
                timestamp: ts,
                flags: bladerf::bladerf_meta_tx::FLAG_TX_BURST_START as u32 | bladerf::bladerf_meta_tx::FLAG_TX_BURST_END as u32,
                status: 0,
                actual_count: 0,
                reserved: [0u8; 32],
            };

            match dev_arc_tx.sync_tx_meta(&tx_data, &mut meta, 20000) {
                Err(v) => println!("error {:} {:}", v, ts),
                Ok(_) => println!("success {:}", ts),
            };

            //println!("tx:timestamp:{:} actual_count:{:}", meta.timestamp, meta.actual_count);

            match tx_rx.recv_timeout(Duration::from_millis(0)) {
                Ok(_) => break,
                Err(_) => (),
            };
        }
    });

    //tx_handler.join().expect("joined tx thread");

    let mut rx_data = vec!(
        Complex::<i16> { re: 0i16, im: 0i16 };
        samps * 3
    );

    let mut rx_signal = vec!(
        Complex::<f64> {
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

    let correlate = FftCorrelate64::new(samps + sample_distance - 1, samps);

    let mut avg_diff = 0f64;
    let mut avg_diff_cnt = 0f64;

    let mut cycle = 0usize;

    let mut avg_buf = vec!(0u32; sample_distance * avg_buf_lines);
    let mut avg_buf2 = vec!(0f64; sample_distance * avg_buf_lines);

    let mut rng = rand::thread_rng();

    //let mut planner = rustfft::FftPlanner::new();
    //let fft_forward = planner.plan_fft(samps, rustfft::FftDirection::Forward);
    //let fft_backward = planner.plan_fft(samps, rustfft::FftDirection::Inverse);

    loop {
        let mut rx_datas: Vec<(usize, Vec<Complex<i16>>)> = Vec::new();
        let mut meta;
        
        meta = bladerf::Struct_bladerf_metadata::default();
        meta.flags = bladerf::bladerf_meta_rx::FLAG_RX_NOW as u32;
        dev_arc.sync_rx_meta(&mut rx_trash, &mut meta, 20000).expect("rx sync call [trash]");

        while rx_datas.len() < avg_buf_lines {
            // Clear the buffer.
            let initial_timestamp = rx_rx.recv().expect("sync tx reply with initial timestamp");

            meta = bladerf::Struct_bladerf_metadata::default();
            meta.timestamp = initial_timestamp;
            match dev_arc.sync_rx_meta(&mut rx_data, &mut meta, 20000) {
                Err(err) => match err {
                    -14 => continue,
                    _ => panic!("sync_rx_meta error {:}", err),
                },
                Ok(_) => (),
            }

            if meta.actual_count as usize != rx_data.len() {
                println!("actual:{:} exp:{:}", meta.actual_count, rx_data.len());
                continue;
            }

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
            convert_iqi16_to_iqf64(&rx_datas[cycle].1, &mut rx_signal);
            // Scale it down so the correlation sums don't get larger than needed.
            div_iqf64_scalar_inplace(&mut rx_signal, 2896.309);

            let mut cor_buffer: Vec<Vec<f64>> = Vec::with_capacity(shifted_copies.len());

            println!("doing correlations");
            for i in 0..shifted_copies.len() {
                let ss = &shifted_copies[i];
                println!("i:{:}/{:}", i, shifted_copies.len() - 1);
                /*
                fn phase_shift_signal_by_distance(
                    sps: f64,
                    center_freq: f64,
                    signal_in: &[Complex<f32>],
                    dist: f64,
                    wave_velocity: f64) -> Vec<Complex<f64>> {
                */

                /*let res = correlate.correlate(
                    &rx_signal[best_ndx..best_ndx + samps + sample_distance],
                    &ss
                );*/
                
                let mut res: Vec<Complex<f64>> = Vec::with_capacity(sample_distance);
                for u in 0..sample_distance {
                    let dist = wave_velocity / sps as f64 * u as f64;
                    /*let shifted_ss = phase_shift_signal_by_distance(
                        sps as f64, freq, ss, dist, wave_velocity,
                        &fft_forward, &fft_backward
                    );*/
                    let shifted_ss = ss;

                    // https://en.wikipedia.org/wiki/Cross-correlation#Zero-normalized_cross-correlation_(ZNCC)
                    // I've tried to do the zero-normalized cross-correlation.

                    let mut avg0 = Complex::<f64>::default();
                    let mut avg1 = Complex::<f64>::default();

                    for w in 0..samps {
                        avg0 += rx_signal[best_ndx + u + w];
                        avg1 += shifted_ss[w];
                    }

                    avg0 /= samps as f64;
                    avg1 /= samps as f64;

                    let mut sd0 = Complex::<f64>::default();
                    let mut sd1 = Complex::<f64>::default();

                    for w in 0..samps {
                        let a = rx_signal[best_ndx + u + w] - avg0;
                        let b = shifted_ss[w] - avg1;
                        sd0 += a * a;
                        sd1 += b * b;
                    }

                    let sd0f = f64::sqrt((sd0 / samps as f64).norm_sqr());
                    let sd1f = f64::sqrt((sd1 / samps as f64).norm_sqr());
    
                    let mut sample = Complex::<f64>::default();
                    for w in 0..samps {
                        sample += 1.0 / (sd0f * sd1f) * (rx_signal[best_ndx + u + w] - avg0) * (shifted_ss[w].conj() - avg1);
                    }
                    res.push(sample / samps as f64);
                }

                let mut out: Vec<f64> = Vec::with_capacity(res.len());
                for y in 0..res.len() {
                    out.push(f64::sqrt(res[y].norm_sqr()));
                }
                cor_buffer.push(out);
            }

            println!("doing max of set");
            for i in 0..sample_distance {
                let mut high_value = 0f64;
                let mut high_index = 0usize;
                for y in 0..cor_buffer.len() {
                    let value = cor_buffer[y][i];
                    if value > high_value {
                        high_value = value;
                        high_index = y;
                    }
                }

                avg_buf[sample_distance * cycle + i] = high_index as u32;
                avg_buf2[sample_distance * cycle + i] = high_value;
            }
        }

        for i in 0..sample_distance {
            let mut avg = 0f64;
            let mut avg2 = 0f64;
            for y in 0..avg_buf_lines {
                avg += avg_buf[sample_distance * y + i] as f64;
                avg2 += avg_buf2[sample_distance * y + i];
            }
            avg /= avg_buf_lines as f64;
            avg2 /= avg_buf_lines as f64;
            // Write the value to the file.
            fout_buffer.seek(SeekFrom::Start(0)).expect("seeking into buffer");
            fout_buffer.write_f32::<LittleEndian>(avg as f32).expect("encoding f64 into bytes");
            fout.write(fout_buffer.get_ref()).expect("writing magnitude to file");
            fout_buffer.seek(SeekFrom::Start(0)).expect("seeking into buffer");
            fout_buffer.write_f32::<LittleEndian>(avg2 as f32).expect("encoding f64 into bytes");
            fout.write(fout_buffer.get_ref()).expect("writing magnitude to file");
        }
    }

    rx_tx.send(true).expect("tried sending tx thread shutdown command");

    println!("joining tx thread");
    tx_handler.join().expect("joining tx thread");
}
