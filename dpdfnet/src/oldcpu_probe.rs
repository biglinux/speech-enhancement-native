//! Standalone, single-thread diagnostic copied UNCHANGED into the frozen control.
//! No Python/NumPy/perf/compiler is required to run this executable on the i3.
//! Benchmark entry sets FTZ/DAZ just like the deployed C/LADSPA entry points.
#![allow(unexpected_cfgs)]
use dpdfnet_native::{AudioProcessor, Bundle, BINS};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{alloc::{GlobalAlloc,Layout,System}, fs::{self,File}, io::{BufReader,BufWriter,Read,Write},
    path::Path, sync::atomic::{AtomicBool,AtomicU64,Ordering}, time::{Duration,Instant}};

struct Audit;
static ARMED:AtomicBool=AtomicBool::new(false);
static ALLOCS:AtomicU64=AtomicU64::new(0);
static FREES:AtomicU64=AtomicU64::new(0);
unsafe impl GlobalAlloc for Audit {
    unsafe fn alloc(&self,l:Layout)->*mut u8 {
        if ARMED.load(Ordering::Relaxed) { ALLOCS.fetch_add(1,Ordering::Relaxed); }
        unsafe { System.alloc(l) }
    }
    unsafe fn alloc_zeroed(&self,l:Layout)->*mut u8 {
        if ARMED.load(Ordering::Relaxed) { ALLOCS.fetch_add(1,Ordering::Relaxed); }
        unsafe { System.alloc_zeroed(l) }
    }
    unsafe fn dealloc(&self,p:*mut u8,l:Layout) {
        if ARMED.load(Ordering::Relaxed) { FREES.fetch_add(1,Ordering::Relaxed); }
        unsafe { System.dealloc(p,l) }
    }
    unsafe fn realloc(&self,p:*mut u8,l:Layout,n:usize)->*mut u8 {
        if ARMED.load(Ordering::Relaxed) { ALLOCS.fetch_add(1,Ordering::Relaxed); }
        unsafe { System.realloc(p,l,n) }
    }
}
#[global_allocator] static ALLOC:Audit=Audit;
fn ftz_daz() -> u32 {
    #[cfg(target_arch="x86_64")]
    unsafe {
        let mut csr=0u32;
        std::arch::asm!("stmxcsr [{p}]",p=in(reg) &mut csr,options(nostack,preserves_flags));
        csr|=0x8040;
        std::arch::asm!("ldmxcsr [{p}]",p=in(reg) &csr,options(nostack,readonly,preserves_flags));
        return csr;
    }
    #[allow(unreachable_code)] 0
}
fn cpu_seconds()->Result<f64,String> {
    #[cfg(target_os="linux")]
    {
        #[repr(C)] struct Ts { sec:std::os::raw::c_long, nsec:std::os::raw::c_long }
        extern "C" { fn clock_gettime(id:std::os::raw::c_int,ts:*mut Ts)->std::os::raw::c_int; }
        let mut t=Ts{sec:0,nsec:0};
        if unsafe { clock_gettime(3,&mut t) } != 0 { return Err("CLOCK_THREAD_CPUTIME_ID failed".into()); }
        Ok(t.sec as f64+t.nsec as f64*1e-9)
    }
    #[cfg(not(target_os="linux"))] Err("Probe CPU-time measurement requires Linux".into())
}
fn environment()->Value {
    let text=fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
    let brand=text.lines().find_map(|l| l.strip_prefix("model name").and_then(|v|v.split_once(':')).map(|(_,v)|v.trim())).unwrap_or("unknown");
    let flags=text.lines().find_map(|l| l.strip_prefix("flags").and_then(|v|v.split_once(':')).map(|(_,v)|v.trim())).unwrap_or("");
    #[cfg(target_arch="x86_64")]
    let (avx,avx2,fma,sse)=(std::is_x86_feature_detected!("avx"),std::is_x86_feature_detected!("avx2"),std::is_x86_feature_detected!("fma"),std::is_x86_feature_detected!("sse4.1"));
    #[cfg(not(target_arch="x86_64"))] let (avx,avx2,fma,sse)=(false,false,false,false);
    #[cfg(target_arch="x86_64")] let tier=dfn_ops::simd_tier();
    #[cfg(not(target_arch="x86_64"))] let tier=0u8;
    let integer=if cfg!(feature="scalar-reference") {"scalar"}
        else if avx2 && !cfg!(feature="force-sse41") && !cfg!(feature="force-avx1") {"avx2"}
        else if sse && avx && cfg!(feature="oldcpu-vex128") && !cfg!(feature="force-sse41") {"vex128"}
        else if sse {"sse41"} else {"scalar"};
    json!({"cpu":brand,"flags":flags,"avx":avx,"avx2":avx2,"fma":fma,"sse41":sse,
        "project_fp_tier":tier,"packed_integer_backend":integer,
        "dependency_dispatch_unforced":true,"mxcsr":ftz_daz(),
        "rational_active":cfg!(feature="oldcpu-rational-gates") && (tier==1||tier==2) && !cfg!(feature="scalar-reference"),
        "features":{"packed_gru":cfg!(feature="packed-gru"),"specialized_conv":cfg!(feature="specialized-conv"),"dprnn_exact":cfg!(feature="dprnn-exact"),
        "int_tiles":cfg!(feature="oldcpu-int-tiles"),"vex128":cfg!(feature="oldcpu-vex128"),"exp_bits":cfg!(feature="oldcpu-exp-bits"),
        "df5_avx":cfg!(feature="oldcpu-df5-avx"),"rational_gates":cfg!(feature="oldcpu-rational-gates"),
        "scalar_reference":cfg!(feature="scalar-reference"),"force_avx1":cfg!(feature="force-avx1"),"force_sse41":cfg!(feature="force-sse41"),
        "r8_packed_tiles":cfg!(feature="r8-packed-tiles"),"r8_fb_pair":cfg!(feature="r8-fb-pair"),
        "r8_sse_prebroadcast":cfg!(feature="r8-sse-prebroadcast"),"r8_matvec_wide":cfg!(feature="r8-matvec-wide"),
        "r8_amax":cfg!(feature="r8-amax")}})
}
fn next(s:&mut u32)->f32 { *s^=*s<<13; *s^=*s>>17; *s^=*s<<5; *s as f32/u32::MAX as f32-0.5 }
fn synthetic(n:usize)->Vec<f32> {
    let mut state=0x71f337a5u32;
    (0..n).map(|i| {
        let phase=i%(48000*7); let noise=next(&mut state);
        if (72000..216000).contains(&phase) { 0.0 }
        else if (216000..240000).contains(&phase) { 1e-7*noise }
        else { let level=if phase>=240000 {0.001} else {0.03};
            level*((i as f32*0.02879793).sin()+0.5*noise) }
    }).collect()
}
fn digest_f32(x:&[f32])->String { let mut s=Sha256::new(); for v in x {s.update(v.to_le_bytes());} format!("{:x}",s.finalize()) }
fn write_f32(path:&Path,x:&[f32])->Result<(),String> {
    let mut f=BufWriter::new(File::create(path).map_err(|e|e.to_string())?);
    for v in x { f.write_all(&v.to_le_bytes()).map_err(|e|e.to_string())?; }
    f.flush().map_err(|e|e.to_string())
}
fn raw(path:&str,n:usize)->Result<Vec<f32>,String> {
    if path=="-" {return Ok(synthetic(n));}
    let meta=fs::metadata(path).map_err(|e|e.to_string())?;
    if !meta.is_file() {return Err("raw input must be a regular file, not a pipe/device".into());}
    let size=meta.len();
    if size==0||size%4!=0||size>256*1024*1024 {return Err("invalid raw f32 input size".into());}
    let bytes=fs::read(path).map_err(|e|e.to_string())?;
    if bytes.is_empty()||bytes.len()%4!=0||bytes.len()>256*1024*1024 {return Err("invalid raw f32 input size".into());}
    let samples:Vec<f32>=bytes.chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())).collect();
    if samples.iter().any(|x|!x.is_finite()) {return Err("non-finite raw input".into());}
    // Repetition is a BENCH workload only; never call the looped audio a quality render.
    Ok((0..n).map(|i|samples[i%samples.len()]).collect())
}
fn percentile(x:&mut [f64])->Value {
    x.sort_by(f64::total_cmp);let n=x.len();
    let q=|p:usize|x[((n*p).div_ceil(100)).saturating_sub(1).min(n-1)];
    json!({"median":q(50),"p95":q(95),"p99":q(99),"max":x[n-1]})
}
fn bench(bundle:&str,secs:usize,quantum:usize,input:&str,paced:bool)->Result<Value,String> {
    if !(1..=600).contains(&secs)||!(1..=8192).contains(&quantum) {return Err("seconds 1..600 and quantum 1..8192 required".into());}
    let b=Bundle::open(bundle)?;let mut p=AudioProcessor::new(b.clone())?;
    let samples=raw(input,secs*48000)?;let hash=digest_f32(&samples);
    let callbacks=samples.len().div_ceil(quantum);
    let mut output=vec![0.0f32;quantum];let mut times=vec![0.0f64;callbacks];
    let mut checksum=0.0f64;let mut misses=0u64;let mut late_starts=0u64;
    let meta=environment();
    let weights_hash=b.manifest()?.get("weights_sha256").and_then(Value::as_str).ok_or("missing weight hash")?.to_owned();
    // Two audio seconds warm BOTH signal and recurrence. Samples include silence later.
    for block in samples[..samples.len().min(96000)].chunks(quantum) {ftz_daz();p.process(block,&mut output[..block.len()]);}
    p.reset();if paced {std::thread::sleep(Duration::from_millis(1));}
    let cpu0=cpu_seconds()?;
    ALLOCS.store(0,Ordering::SeqCst);FREES.store(0,Ordering::SeqCst);
    ARMED.store(true,Ordering::SeqCst);
    let origin=Instant::now();let mut work=0.0f64;
    for (i,block) in samples.chunks(quantum).enumerate() {
        let release=origin+Duration::from_secs_f64((i*quantum) as f64/48000.0);
        if paced {let now=Instant::now();if now<release {std::thread::sleep(release-now);}}
        let start=Instant::now();
        if paced && start.saturating_duration_since(release).as_micros()>100 {late_starts+=1;}
        ftz_daz();p.process(block,&mut output[..block.len()]);
        let end=Instant::now();let dt=end.duration_since(start).as_secs_f64();times[i]=dt*1e6;work+=dt;
        checksum+=output[block.len()/2] as f64;
        if paced && end>release+Duration::from_secs_f64(block.len() as f64/48000.0) {misses+=1;}
    }
    let elapsed=origin.elapsed().as_secs_f64();ARMED.store(false,Ordering::SeqCst);
    let cpu=cpu_seconds()?-cpu0;
    let alloc=ALLOCS.load(Ordering::Relaxed);let frees=FREES.load(Ordering::Relaxed);
    let good=alloc==0&&frees==0&&!p.faulted()&&p.sanitized_samples==0&&p.hops==secs as u64*100&&checksum.is_finite();
    Ok(json!({"environment":meta,"mode":if paced {"paced"} else {"throughput"},"audio_seconds":secs,
        "quantum":quantum,"callbacks":callbacks,"processed_hops":p.hops,"fault":p.faulted(),"sanitized_samples":p.sanitized_samples,
        "hot_allocations":alloc,"hot_deallocations":frees,"integrity_passed":good,
        "work_rtf":work/secs as f64,"wall_rtf":elapsed/secs as f64,"thread_cpu_rtf":cpu/secs as f64,
        "callback_us":percentile(&mut times),"callback_budget_us":quantum as f64/48000.0*1e6,
        "deadline_misses":misses,"late_starts_over_100us":late_starts,"checksum":checksum,"input_sha256":hash,"input_kind":if input=="-" {"synthetic"} else {"recording"},
        "weights_bytes":b.weight_bytes(),"weights_sha256":weights_hash,"note":"No realtime scheduling acquired here. Paced includes scheduler interference; throughput is not live PipeWire certification."}))
}
fn capture(bundle:&str,dir:&Path,frames:usize)->Result<Value,String> {
    if !(8..=4096).contains(&frames)||dir.exists() {return Err("new output directory and 8..4096 frames required".into());}
    fs::create_dir_all(dir).map_err(|e|e.to_string())?;
    let b=Bundle::open(bundle)?;let mut p=AudioProcessor::new(b)?;let env=environment();
    let names=["magnitude_features","complex_features","erb0","erb1","erb2","erb3","df0","df1","erb_dual","df_dual","embedding","mask","coefficients","spectrum"];
    let mut buffers:Vec<Vec<f32>>=names.iter().map(|_|Vec::new()).collect();
    let mut state=0x17451733u32;
    for n in 0..frames {
        if n==frames/2 {p.reset();}
        let level=if n<4 {0.0} else if n<frames/3 {0.00001} else {0.08};
        let mut s=[0.0f32;BINS*2];for x in &mut s {*x=next(&mut state)*level;}s[1]=0.0;s[BINS*2-1]=0.0;
        p.model.process_spectrum(&s,0.0);
        for i in 0..names.len() {buffers[i].extend_from_slice(p.model.trace(i).ok_or("missing trace")?);}
    }
    let mut files=Vec::new();
    for (name,v) in names.iter().zip(buffers.iter()) {
        if v.iter().any(|x|!x.is_finite()) {return Err(format!("non-finite trace {name}"));}
        let file=format!("trace_{name}.f32");write_f32(&dir.join(&file),v)?;
        files.push(json!({"file":file,"elements":v.len(),"sha256":digest_f32(v)}));
    }
    let samples=synthetic(8*48000);let mut whole=vec![0.0;samples.len()];p.reset();p.process(&samples,&mut whole);
    let mut variable=vec![0.0;samples.len()];p.reset();let sizes=[1,7,64,128,256,480,512,960,1024];let mut pos=0;let mut cycle=0;
    while pos<samples.len() {let end=(pos+sizes[cycle%sizes.len()]).min(samples.len());p.process(&samples[pos..end],&mut variable[pos..end]);pos=end;cycle+=1;}
    if whole.iter().map(|v|v.to_bits()).ne(variable.iter().map(|v|v.to_bits())) {return Err("block partition/reset mismatch".into());}
    if p.faulted()||whole.iter().any(|v|!v.is_finite()) {return Err("audio capture fault".into());}
    for (name,v) in [("pcm.f32",&whole),("pcm_variable.f32",&variable)] {
        write_f32(&dir.join(name),v)?;files.push(json!({"file":name,"elements":v.len(),"sha256":digest_f32(v)}));
    }
    let result=json!({"environment":env,"frames":frames,"files":files,"partition_reset_passed":true});
    fs::write(dir.join("capture.json"),serde_json::to_vec_pretty(&result).unwrap()).map_err(|e|e.to_string())?;Ok(result)
}
fn matrices(v:&Value,out:&mut Vec<Value>) {
    if v.get("kind").and_then(Value::as_str)==Some("w8a16")&&v.get("rows").is_some(){out.push(v.clone());return;}
    match v {Value::Object(m)=>for x in m.values(){matrices(x,out)},Value::Array(a)=>for x in a {matrices(x,out)},_=>{}}
}
fn micro(bundle:&str,iters:usize)->Result<Value,String> {
    if !(10..=100000).contains(&iters){return Err("micro iterations 10..100000".into());}
    let b=Bundle::open(bundle)?;let env=environment();let mut vs=Vec::new();matrices(&b.manifest()?,&mut vs);
    let mut report=Vec::new();let mut seed=0x7532155u32;
    for width in [64usize,256] {
        let v=vs.iter().find(|v|v["cols"].as_u64()==Some(width as u64)&&v["rows"].as_u64()==Some((3*width) as u64)).ok_or("missing matrix geometry")?;
        let m=dpdfnet_native::kernels::Matrix::load(&b,v)?;
        for count in [1usize,4] {
            let x:Vec<f32>=(0..width*count).map(|_|next(&mut seed)).collect();let mut y=vec![0.0;3*width*count];let mut q=vec![0;4*width];
            let begin=Instant::now();for _ in 0..iters {m.batch(std::hint::black_box(&x),&mut y,count,&mut q);std::hint::black_box(&y);}
            report.push(json!({"operation":"matrix","cols":width,"rows":3*width,"count":count,"us_per_call":begin.elapsed().as_secs_f64()*1e6/iters as f64}));
        }
        let mut h:Vec<f32>=(0..width).map(|_|next(&mut seed)).collect();let wx:Vec<f32>=(0..width*3).map(|_|next(&mut seed)).collect();let rh=wx.clone();let bias=vec![0.02;6*width];
        let begin=Instant::now();for _ in 0..iters {dfn_ops::gru_update(std::hint::black_box(&mut h),&wx,&rh,&bias);}
        report.push(json!({"operation":"gates","hidden":width,"us_per_call":begin.elapsed().as_secs_f64()*1e6/iters as f64}));
    }
    // RealFFT already uses a real-input transform. Time its ACTUAL detected backend.
    let mut planner=realfft::RealFftPlanner::<f32>::new();
    let fwd=planner.plan_fft_forward(960);let inv=planner.plan_fft_inverse(960);
    let mut real=vec![0.0f32;960];let mut freq=fwd.make_output_vec();
    let mut scratch_f=fwd.make_scratch_vec();let mut scratch_i=inv.make_scratch_vec();
    let original:Vec<f32>=(0..960).map(|_|next(&mut seed)).collect();
    let start=Instant::now();
    for _ in 0..iters {
        real.copy_from_slice(&original);
        fwd.process_with_scratch(&mut real,&mut freq,&mut scratch_f).map_err(|e|e.to_string())?;
        inv.process_with_scratch(&mut freq,&mut real,&mut scratch_i).map_err(|e|e.to_string())?;
        std::hint::black_box(&real);
    }
    report.push(json!({"operation":"realfft_pair_960_with_input_restore","us_per_pair":start.elapsed().as_secs_f64()*1e6/iters as f64,
       "dependency_dispatch_unforced":true}));
    Ok(json!({"environment":env,"iterations":iters,"microbenchmarks":report,"note":"Synthetic inputs, not a model self-time profile or old-CPU speedup certificate."}))
}
fn integer_selftest()->Result<Value,String> {
    let mut cases=0usize;
    for n in [2usize,6,16,64,256,1024] {for m in [8usize,24,32,64,72,192,768] {for extreme in [false,true] {
        let weights:Vec<i8>=(0..m*n).map(|i|if extreme {127} else {((i*71+13)%255) as i16 as i8}).map(|v|if v==-128 {0}else{v}).collect();
        let mut bytes=vec![0u8;m*n];
        for r in 0..m {for j in 0..n {bytes[(r/8)*(8*n)+(j/2)*16+(r%8)*2+j%2]=weights[r*n+j] as u8;}}
        while bytes.len()%64!=0 {bytes.push(0);}
        let offset=bytes.len();let scales:Vec<f32>=(0..m).map(|r|0.0001*(r%13+1) as f32).collect();for s in &scales {bytes.extend_from_slice(&s.to_le_bytes());}
        let manifest=json!({"schema":2,"architecture":"dpdfnet-48hr-v1","weight_bytes":bytes.len(),"weights_sha256":format!("{:x}",Sha256::digest(&bytes))});
        let b=Bundle::from_bytes(&serde_json::to_vec(&manifest).unwrap(),&bytes)?;
        let mv=json!({"kind":"w8a16","rows":m,"cols":n,"layout":"pair-output8-v1","weight":{"dtype":"i8","offset":0,"len":m*n},"scales":{"dtype":"f32","offset":offset,"len":m}});
        let matrix=dpdfnet_native::kernels::Matrix::load(&b,&mv)?;
        for count in [1usize,4] {
            let x:Vec<f32>=(0..n*count).map(|i|if extreme {1.0} else {((i*193%32767) as f32-16383.0)/32768.0}).collect();
            let mut q=vec![0i16;n*4];let mut out=vec![0.0;m*count];matrix.batch(&x,&mut out,count,&mut q);
            for k in 0..count {
                let scale=dfn_ops::quantize_i16(&x[k*n..(k+1)*n],&mut q[..n]);
                for r in 0..m {let sum:i64=(0..n).map(|j|weights[r*n+j] as i64*q[j] as i64).sum();
                    let expected=(sum as f32*scales[r])*scale;
                    if expected.to_bits()!=out[k*m+r].to_bits(){return Err(format!("integer mismatch rows={m} cols={n} k={k} row={r}"));}
                }
            }
            cases+=1;
        }
    }}}
    Ok(json!({"environment":environment(),"passed":true,"cases":cases,"contract":"selected packed kernels versus independent INT64 sum; no trained weights"}))
}
fn read_f32_limited(path:&Path,bytes:usize)->Result<Vec<f32>,String> {
    if bytes>64*1024*1024||fs::metadata(path).map_err(|e|e.to_string())?.len()!=bytes as u64 {return Err("oracle file length/bound mismatch".into());}
    let data=fs::read(path).map_err(|e|e.to_string())?;
    let x:Vec<f32>=data.chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())).collect();
    if x.iter().any(|v|!v.is_finite()){return Err("non-finite oracle data".into());}Ok(x)
}
fn oracle(bundle:&str,dir:&Path)->Result<Value,String> {
    let path=dir.join("oracle.json");
    if fs::metadata(&path).map_err(|e|e.to_string())?.len()>1024*1024 {return Err("oracle metadata too large".into());}
    let meta:Value=serde_json::from_slice(&fs::read(path).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
    let frames=meta["frames"].as_u64().ok_or("missing oracle frames")? as usize;
    if meta["schema"]!=1 || !(8..=256).contains(&frames) || meta["atol"].as_f64()!=Some(5e-4) || meta["rtol"].as_f64()!=Some(5e-4) {return Err("oracle schema/frame/tolerance contract mismatch".into());}
    if meta["reference"].as_str()!=Some("independent W8A16 oracle") {return Err("expected independent W8A16 oracle".into());}
    let b=Bundle::open(bundle)?;
    if b.manifest()?["weights_sha256"]!=meta["weights_sha256"] {return Err("oracle bundle mismatch".into());}
    let mut p=AudioProcessor::new(b)?;
    let names=["inputs","magnitude_features","complex_features","erb0","erb1","erb2","erb3","df0","df1","erb_dual","df_dual","embedding","mask","coefficients","spectrum"];
    let files=meta["files"].as_array().ok_or("missing oracle files")?;
    if files.len()!=names.len(){return Err("oracle file count mismatch".into());}
    let mut all=Vec::new();let mut widths=Vec::new();let mut total=0usize;
    for name in names {
        let file=format!("{name}.f32");
        let desc=files.iter().find(|v|v["file"].as_str()==Some(file.as_str())).ok_or("missing oracle trace")?;
        let width=desc["per_frame"].as_u64().ok_or("missing trace width")? as usize;
        let n=width.checked_mul(frames).ok_or("oracle size overflow")?;
        let bytes=n.checked_mul(4).ok_or("oracle size overflow")?;
        total=total.checked_add(bytes).ok_or("oracle size overflow")?;
        if total>256*1024*1024||desc["elements"].as_u64()!=Some(n as u64) {return Err("oracle bounds mismatch".into());}
        let values=read_f32_limited(&dir.join(file),bytes)?;
        if desc["sha256"].as_str()!=Some(digest_f32(&values).as_str()){return Err("oracle checksum mismatch".into());}
        if name=="inputs" && width!=BINS*2 {return Err("oracle input width mismatch".into());}
        widths.push(width);all.push(values);
    }
    let mut max_abs=[0.0f64;14];let mut max_relative=[0.0f64;14];let mut violations=[0usize;14];let mut first=Value::Null;
    for frame in 0..frames {
        ftz_daz();p.model.process_spectrum(&all[0][frame*962..(frame+1)*962],0.0);
        for id in 0..14 {
            let got=p.model.trace(id).ok_or("missing native trace")?;
            if got.len()!=widths[id+1]{return Err(format!("oracle trace size mismatch id={id}"));}
            let want=&all[id+1][frame*got.len()..(frame+1)*got.len()];
            let mut e2=0.0;let mut r2=0.0;
            for (&x,&y) in got.iter().zip(want) {
                let error=(x as f64-y as f64).abs();
                if !x.is_finite() {return Err(format!("non-finite native oracle trace id={id}"));}
                max_abs[id]=max_abs[id].max(error);e2+=error*error;r2+=(y as f64)*(y as f64);
                if error>5e-4+5e-4*(y as f64).abs(){violations[id]+=1;if first.is_null(){first=json!({"frame":frame,"trace":id,"name":names[id+1]});}}
            }
            max_relative[id]=max_relative[id].max(e2.sqrt()/(r2.sqrt()+1e-12));
        }
    }
    let layers:Vec<Value>=(0..14).map(|i|json!({"name":names[i+1],"max_abs":max_abs[i],"max_relative_rms":max_relative[i],"violations":violations[i],"passed":violations[i]==0})).collect();
    Ok(json!({"passed":first.is_null(),"first_failure":first,"layers":layers,"frames":frames,"atol":5e-4,"rtol":5e-4,
      "environment":environment(),"reference":"independent W8A16 oracle exported on build host","weights_sha256":meta["weights_sha256"],
      "quality_approved":false,"note":"No inherited-failure waiver. Reference generated once; candidate evolves its own recurrent state."}))
}


/// Offline renderer, not a benchmark. Same AudioProcessor core as the C ABI, no ctypes,
/// pipe I/O, per-frame reset, repeated input, independent gain alignment or flush.
fn save_json(path: &Path, value: &Value) -> Result<(), String> {
    let temp=path.with_extension("json.partial");
    fs::write(&temp,serde_json::to_vec_pretty(value).map_err(|e|e.to_string())?)
        .map_err(|e|e.to_string())?;
    fs::rename(temp,path).map_err(|e|e.to_string())
}
fn render(bundle: &str, input: &Path, dir: &Path, db: f32) -> Result<Value,String> {
    if !db.is_finite() || !(0.0..=100.0).contains(&db) {return Err("db must be finite in 0..100".into());}
    let meta=fs::metadata(input).map_err(|e|e.to_string())?;
    if !meta.is_file() || meta.len()==0 || meta.len()%4!=0 || meta.len()>512*1024*1024 {
        return Err("render input must be a nonempty bounded REGULAR raw float32 file".into());
    }
    if dir.exists() {return Err("render output must be a NEW directory".into());}
    fs::create_dir_all(dir).map_err(|e|e.to_string())?;
    let progress=dir.join("progress.json");
    save_json(&progress,&json!({"phase":"loading-model","samples":0,"total_samples":meta.len()/4}))?;
    eprintln!("render: loading model; progress={}",progress.display());
    let result=(|| -> Result<Value,String> {
        let b=Bundle::open(bundle)?;
        let weight_hash=b.manifest()?["weights_sha256"].clone();
        let mut p=AudioProcessor::new(b)?;p.set_attenuation_db(db);
        let env=environment();
        save_json(&progress,&json!({"phase":"rendering","samples":0,"total_samples":meta.len()/4}))?;
        let mut inp=BufReader::new(File::open(input).map_err(|e|e.to_string())?);
        let partial=dir.join("pcm.f32.partial");
        let mut writer=BufWriter::new(File::create(&partial).map_err(|e|e.to_string())?);
        let mut raw=[0u8;960*4];let mut pcm=[0.0f32;960];let mut output=[0.0f32;960];
        let mut input_hash=Sha256::new();let mut output_hash=Sha256::new();
        let mut remaining=meta.len() as usize;let mut done=0usize;let mut report_at=0usize;let mut peak=0.0f32;
        while remaining>0 {
            let bytes=remaining.min(raw.len());
            inp.read_exact(&mut raw[..bytes]).map_err(|e|e.to_string())?;
            input_hash.update(&raw[..bytes]);
            for (x,b) in pcm.iter_mut().zip(raw[..bytes].chunks_exact(4)) {
                *x=f32::from_le_bytes(b.try_into().unwrap());
                if !x.is_finite() { return Err("non-finite render input".into()); }
            }
            let count=bytes/4;
            ftz_daz();p.process(&pcm[..count],&mut output[..count]);
            if p.faulted() || p.sanitized_samples!=0 {return Err("native render fault".into());}
            for (x,b) in output[..count].iter().zip(raw[..bytes].chunks_exact_mut(4)) {
                if !x.is_finite(){return Err("non-finite render output".into());}
                peak=peak.max(x.abs());b.copy_from_slice(&x.to_le_bytes());
            }
            writer.write_all(&raw[..bytes]).map_err(|e|e.to_string())?;
            output_hash.update(&raw[..bytes]);done+=count;remaining-=bytes;
            // Early first-block flush makes a slow first inference distinguishable
            // from no output buffering; later progress once per audio second.
            if done>=report_at {
                writer.flush().map_err(|e|e.to_string())?;
                save_json(&progress,&json!({"phase":"rendering","samples":done,"total_samples":meta.len()/4,"hops":p.hops}))?;
                eprintln!("render: samples={done}/{} hops={}",meta.len()/4,p.hops);
                report_at=done+48000;
            }
        }
        let mut extra=[0u8;1];
        if inp.read(&mut extra).map_err(|e|e.to_string())?!=0 {return Err("input changed length during render".into());}
        writer.flush().map_err(|e|e.to_string())?;drop(writer);
        if fs::metadata(&partial).map_err(|e|e.to_string())?.len()!=meta.len() {return Err("incomplete output length".into());}
        let pcm_path=dir.join("pcm.f32");fs::rename(partial,&pcm_path).map_err(|e|e.to_string())?;
        let out_hash=format!("{:x}",output_hash.finalize());
        let in_hash=format!("{:x}",input_hash.finalize());
        let mut wav=BufWriter::new(File::create(dir.join("audio.wav.partial")).map_err(|e|e.to_string())?);
        // Float32 WAVE: same latency/gain/duration as native control; no dithering.
        let mut header=Vec::with_capacity(44);
        header.extend_from_slice(b"RIFF");header.extend_from_slice(&((meta.len()+36) as u32).to_le_bytes());
        header.extend_from_slice(b"WAVEfmt ");header.extend_from_slice(&16u32.to_le_bytes());
        header.extend_from_slice(&3u16.to_le_bytes());header.extend_from_slice(&1u16.to_le_bytes());
        header.extend_from_slice(&48000u32.to_le_bytes());header.extend_from_slice(&192000u32.to_le_bytes());
        header.extend_from_slice(&4u16.to_le_bytes());header.extend_from_slice(&32u16.to_le_bytes());
        header.extend_from_slice(b"data");header.extend_from_slice(&(meta.len() as u32).to_le_bytes());
        wav.write_all(&header).map_err(|e|e.to_string())?;
        std::io::copy(&mut File::open(&pcm_path).map_err(|e|e.to_string())?,&mut wav).map_err(|e|e.to_string())?;
        wav.flush().map_err(|e|e.to_string())?;drop(wav);
        fs::rename(dir.join("audio.wav.partial"),dir.join("audio.wav")).map_err(|e|e.to_string())?;
        save_json(&dir.join("capture.json"),&json!({"files":[{"file":"pcm.f32","elements":done,"sha256":out_hash}]}))?;
        Ok(json!({"passed":true,"environment":env,"input_sha256":in_hash,"pcm_sha256":out_hash,
            "samples":done,"hops":p.hops,"peak":peak,"attenuation_db":db,"weights_sha256":weight_hash,
            "latency_samples":dpdfnet_native::audio::LATENCY,"quality_approved":false,
            "policy":"same-length causal render; no looping or independent realignment"}))
    })();
    match &result {
        Ok(value)=>{save_json(&dir.join("render.json"),value)?;save_json(&progress,&json!({"phase":"complete","samples":meta.len()/4}))?;}
        Err(error)=>{let _=save_json(&dir.join("render.json"),&json!({"passed":false,"error":error}));let _=save_json(&progress,&json!({"phase":"failed","error":error}));}
    }
    result
}

fn parse(a:Option<&String>,default:usize)->Result<usize,String>{a.map_or(Ok(default),|s|s.parse::<usize>().map_err(|e|e.to_string()))}
fn go()->Result<Value,String> {
    let a:Vec<String>=std::env::args().collect();ftz_daz();
    match a.get(1).map(String::as_str) {
        Some("info")=>Ok(environment()),
        Some("render") if (5..=6).contains(&a.len()) =>render(&a[2],Path::new(&a[3]),Path::new(&a[4]),
            a.get(5).map_or(Ok(100.0),|s|s.parse::<f32>().map_err(|e|e.to_string()))?),
        Some("selftest")=>integer_selftest(),
        Some("oracle") if a.len()==4=>oracle(&a[2],Path::new(&a[3])),
        Some("capture") if (4..=5).contains(&a.len()) =>capture(&a[2],Path::new(&a[3]),parse(a.get(4),64)?),
        Some("bench") if (3..=7).contains(&a.len()) => {
            let mode=a.get(6).map_or("throughput",String::as_str);
            if mode!="throughput" && mode!="paced" {return Err("mode must be throughput or paced".into());}
            bench(&a[2],parse(a.get(3),30)?,parse(a.get(4),960)?,a.get(5).map_or("-",String::as_str),mode=="paced")
        },
        Some("micro") if (3..=4).contains(&a.len()) =>micro(&a[2],parse(a.get(3),1000)?),
        _=>Err("usage: oldcpu_probe render BUNDLE INPUT_F32 NEW_DIR [DB] | info | selftest | oracle BUNDLE GOLDEN_DIR | capture BUNDLE NEW_DIR [FRAMES] | bench BUNDLE [SECONDS] [QUANTUM] [RAW_F32_OR_-] [throughput|paced] | micro BUNDLE [ITERS]".into())
    }
}
fn main(){match go(){Ok(v)=>{println!("{}",serde_json::to_string_pretty(&v).unwrap());if v.get("integrity_passed")==Some(&Value::Bool(false)) || v.get("passed")==Some(&Value::Bool(false)){std::process::exit(2);}},Err(e)=>{eprintln!("{e}");std::process::exit(1);}}}
