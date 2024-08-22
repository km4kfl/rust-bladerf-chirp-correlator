use bladerf;
use std::vec::Vec;
use std::sync::Arc;
use std::ops::{Add, Mul};
use std::thread;
use num_complex::Complex;
use std::sync::mpsc::channel;
use std::time::{Instant, Duration};
use std::fs::File;
use byteorder::{LittleEndian, WriteBytesExt};
use std::io::{Cursor, Write, Seek, SeekFrom};
use rustfft;

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

fn linspace(start: f64, end: f64, steps: u32) -> Vec<f64> {
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

fn roll_slice_left(a: &mut [Complex<f64>]) {
    a[a.len() - 1] = a[0];
    for x in 0..a.len() - 1 {
        a[x] = a[x+1];
    }
}

fn multiply_slice_offset_wrapping_sum(
    a: &[Complex<f64>],
    b: &[Complex<f64>],
    offset: usize) -> Complex<f64> {
    let mut sum = Complex::<f64> { re: 0.0, im: 0.0 };
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

fn conjugate_slice(a: &mut [Complex<f64>]) {
    for va in a.iter_mut() {
        *va = va.conj();
    }
}

fn sum_slice(a: &[Complex<f64>]) -> Complex<f64> {
    let mut out = Complex::<f64> {
        re: 0.0,
        im: 0.0,
    };

    for va in a.iter() {
        out.re += va.re;
        out.im += va.im;
    }

    out
}

fn convert_iqi16_to_iqf64(a: &[Complex<i16>], out: &mut [Complex<f64>]) {
    for (av, out_v) in a.iter().zip(out.iter_mut()) {
        (*out_v).re = av.re as f64;
        (*out_v).im = av.im as f64;
    }
}

fn add_sf64_scalar(a: &[f64], b: f64) -> Vec<f64> {
    let mut out: Vec<f64> = Vec::with_capacity(a.len());
    for v in a.iter() {
        out.push(v + b);
    }

    out
}

fn mul_sf64_scalar(a: &[f64], b: f64) -> Vec<f64> {
    let mut out: Vec<f64> = Vec::with_capacity(a.len());
    for v in a.iter() {
        out.push(v * b);
    }

    out
}

fn fftfreq(sz: usize) -> Vec<f64> {
    let mut out: Vec<f64> = Vec::with_capacity(sz);
    let spacing = 1.0 / sz as f64;
    
    out.push(0.0);

    if sz % 2 != 0 {
        let g = ((sz - 1) / 2) as u32;
        let a = mul_sf64_scalar(&linspace(1.0, g as f64, g), spacing);
        let b = mul_sf64_scalar(&linspace(-(g as f64), -1.0, g), spacing);
        for v in b.iter() {
            out.push(*v);
        }        
        for v in a.iter() {
            out.push(*v);
        }
    } else {
        let g = (sz / 2) as u32;
        let a = mul_sf64_scalar(&linspace(1.0, g as f64 - 1.0, g - 1), spacing);
        let b = mul_sf64_scalar(&linspace(-(g as f64), -1.0, g), spacing);
        for v in b.iter() {
            out.push(*v);
        }        
        for v in a.iter() {
            out.push(*v);
        }
    }

    out
}

fn phase_shift_signal_by_distance(
    sps: f64,
    center_freq: f64,
    signal_in: &[Complex<f64>],
    dist: f64,
    wave_velocity: f64) -> Vec<Complex<f64>> {
    let mut planner = rustfft::FftPlanner::new();
    let fft_forward = planner.plan_fft(signal_in.len(), rustfft::FftDirection::Forward);
    let fft_backward = planner.plan_fft(signal_in.len(), rustfft::FftDirection::Inverse);
    
    let mut signal = Vec::with_capacity(signal_in.len());
    signal.extend_from_slice(signal_in);

    fft_forward.process(&mut signal);

    let freqs = fftfreq(signal.len());
    
    for i in 0..freqs.len() {
        let phase_distance = dist / (wave_velocity / (freqs[i] * sps * 0.5 + center_freq)) * std::f64::consts::PI * 2.0;
        signal[i] *= (Complex::<f64> {
            re: 0.0,
            im: phase_distance,
        }).exp();
    }

    fft_backward.process(&mut signal);

    signal
}

struct FftCorrelate {
    fft_size: usize,
    planner: rustfft::FftPlanner<f64>,
    forward: Arc<dyn rustfft::Fft<f64>>,
    inverse: Arc<dyn rustfft::Fft<f64>>,
}

impl FftCorrelate {
    fn new(a_size: usize, b_size: usize) -> FftCorrelate {
        let fft_size = a_size + b_size + 1;
        let mut planner = rustfft::FftPlanner::new();
        let forward = planner.plan_fft(fft_size, rustfft::FftDirection::Forward);
        let inverse = planner.plan_fft(fft_size, rustfft::FftDirection::Inverse);
        FftCorrelate {
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
            out.push(ap[(adj + x) % ap.len()]);
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

    let freq = 1_830_000_000u64;
    let sps = 520834u32;
    //let sps = 4_000_000u32;
    //let sps = 61_440_000u32;

    dev.set_frequency(bladerf::bladerf_module::RX0, freq).expect(
        "should have set the RX frequency"
    );
    dev.set_frequency(bladerf::bladerf_module::TX0, freq).expect(
        "should have set the TX frequency"
    );

    dev.set_gain(bladerf::bladerf_module::TX0, 10).expect("TX0 gain set");
    dev.set_gain(bladerf::bladerf_module::TX1, 10).expect("TX1 gain set");
    dev.set_gain(bladerf::bladerf_module::RX0, 60).expect("RX0 gain set");
    dev.set_gain(bladerf::bladerf_module::RX1, 60).expect("RX1 gain set");

    dev.set_sample_rate(bladerf::bladerf_module::RX0, 520834).expect(
        "RX0/RX1 sampling rate set"
    );

    dev.set_sample_rate(bladerf::bladerf_module::TX0, 520834).expect(
        "TX0/TX1 sampling rate set"
    );

    let num_buffers = 16u32;
    let buffer_size = 4096u32;
    let num_transfers = 8u32;
    let stream_timeout = 20u32;

    let buffer_samps = num_buffers * buffer_size / 4;

    dev.sync_config(
        bladerf::bladerf_channel_layout::RX_X1,
        bladerf::bladerf_format::SC16_Q11,
        num_buffers, buffer_size,
        Some(num_transfers),
        stream_timeout
    ).expect("sync_config for rx");

    dev.sync_config(
        bladerf::bladerf_channel_layout::TX_X1,
        bladerf::bladerf_format::SC16_Q11,
        num_buffers, buffer_size,
        Some(num_transfers),
        stream_timeout
    ).expect("sync_config for tx");

    dev.enable_module(bladerf::bladerf_module::RX0, true).expect(
        "rx0 module enable"
    );
    dev.enable_module(bladerf::bladerf_module::TX0, true).expect(
        "tx0 module enable"
    );

    let samps: usize = 130208;
    let mut signal: Vec<Complex<f64>> = vec!(Complex::<f64> { re: 0.0f64, im: 0.0f64 }; samps);
    let mut tx_data: Vec<Complex<i16>> = vec!(Complex::<i16> { re: 0i16, im: 0i16 }; samps);

    let freq_start = 10e3f64;
    let freq_end = 100e3f64;
    let theta_step_start = freq_start * std::f64::consts::PI * 2.0f64 / (sps as f64);
    let theta_step_end = freq_end * std::f64::consts::PI * 2.0f64 / (sps as f64);
    let t_space = linspace(theta_step_start, theta_step_end, samps as u32);

    {
        let mut theta = 0.0f64;
        for x in 0..samps {
            signal[x].re = f64::cos(theta);
            signal[x].im = f64::sin(theta);
            tx_data[x].re = (signal[x].re * 2000.0) as i16;
            tx_data[x].im = (signal[x].im * 2000.0) as i16;
            theta += t_space[x];
        }
    }

    let sample_distance = 1000usize;

    let mut signal_slots: Vec<Vec<Complex<f64>>> = Vec::new();
    let wave_velocity = 299792458.0f64;
    let distance_per_sample = wave_velocity / sps as f64;

    for x in 0..sample_distance {
        println!("building cor {:}", x);
        let distance = distance_per_sample * x as f64;
        let mut signal_shifted = phase_shift_signal_by_distance(
            sps as f64,
            freq as f64,
            &signal,
            distance,
            wave_velocity
        );
        conjugate_slice(&mut signal_shifted);
        signal_slots.push(signal_shifted);
    }
    
    let dev_arc = Arc::new(dev);

    let dev_arc_tx = dev_arc.clone();

    let (rx_tx, tx_rx) = channel::<bool>();

    let tx_handler = thread::spawn(move || {
        loop {
            dev_arc_tx.sync_tx(&tx_data, None, 20000).expect("tx sync call");
            match tx_rx.recv_timeout(Duration::from_millis(0)) {
                Ok(_) => break,
                Err(_) => (),
            };
        }
    });

    let mut rx_data: Vec<Complex<i16>> = vec!(
        Complex::<i16> { re: 0i16, im: 0i16 };
        samps * 2
    );

    let mut rx_signal: Vec<Complex<f64>> = vec!(
        Complex::<f64> {
            re: 0.0,
            im: 0.0,
        }; rx_data.len()
    );

    let mut rx_trash: Vec<Complex<i16>> = vec!(
        Complex::<i16> {
            re: 0i16,
            im: 0i16,
        }; buffer_samps as usize
    );

    let st = Instant::now();

    let mut fout = File::create("out.bin").expect("opening output file");
    let mut fout_buffer = Cursor::new(vec!(0u8; 8));

    let correlate = FftCorrelate::new(rx_data.len(), samps);

    for x in 0..1000000 {
        // clear the buffer
        dev_arc.sync_rx(&mut rx_trash, None, 20000).expect("rx sync call [trash]");
        dev_arc.sync_rx(&mut rx_data, None, 20000).expect("rx sync call [data]");
        convert_iqi16_to_iqf64(&rx_data, &mut rx_signal);

        println!("processing");

        let initial_cor = correlate.correlate(&rx_signal, &signal);

        let mut best_mag = 0f64;
        let mut best_ndx = 0usize;

        for i in 0..initial_cor.len() {
            let mag = f64::sqrt(initial_cor[i].norm_sqr());
            if mag > best_mag {
                best_mag = mag;
                best_ndx = i;
            }
        }

        if best_mag < 79_323_227_631_870.28 {
            println!("mag too low");
            continue;
        }

        println!("best_ndx:{:} best_mag:{:}", best_ndx, best_mag);

        for i in 0..signal_slots.len() {
            let ss = &signal_slots[i];
            let mag = f64::sqrt(multiply_slice_offset_wrapping_sum(
                &ss, &rx_signal, best_ndx + i
            ).norm_sqr());
            fout_buffer.seek(SeekFrom::Start(0)).expect("seeking into buffer");
            fout_buffer.write_f64::<LittleEndian>(mag).expect("encoding f64 into bytes");
            fout.write(fout_buffer.get_ref()).expect("writing magnitude to file");                
        }

        if x % 5 == 4 {
            let elapsed = st.elapsed();
            let seconds = elapsed.as_secs() as f64 + (elapsed.subsec_micros() as f64 / 1000000.0);
            let total_samps = seconds * sps as f64;
            if total_samps > x as f64 * samps as f64 {
                println!("WARNING: RX running too slow; losing samples");
            }
        }
    }

    rx_tx.send(true).expect("tried sending tx thread shutdown command");

    println!("joining tx thread");
    tx_handler.join().expect("joining tx thread");
}
