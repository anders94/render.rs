//! Intel Open Image Denoise, loaded at runtime via dlopen (no link-time
//! dependency: machines without the library fall back to the built-in
//! à-trous filter). Same policy as the à-trous path: OIDN filters the
//! DIFFUSE layer with albedo+normal guides; the raw specular layer is
//! added back afterward so refracted detail survives.

use crate::math::Vec3;
use crate::output::film::Film;
use crate::output::Image;
use std::ffi::{c_char, c_void, CStr, CString};
use std::sync::OnceLock;

// dlopen lives in libSystem; no libc crate needed.
extern "C" {
    fn dlopen(path: *const c_char, flag: i32) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}
const RTLD_NOW: i32 = 2;

const OIDN_FORMAT_FLOAT3: i32 = 3;

type NewDeviceFn = unsafe extern "C" fn(i32) -> *mut c_void;
type CommitDeviceFn = unsafe extern "C" fn(*mut c_void);
type NewFilterFn = unsafe extern "C" fn(*mut c_void, *const c_char) -> *mut c_void;
type SetSharedFilterImageFn = unsafe extern "C" fn(
    *mut c_void,
    *const c_char,
    *mut c_void,
    i32,
    usize,
    usize,
    usize,
    usize,
    usize,
);
type SetFilterBoolFn = unsafe extern "C" fn(*mut c_void, *const c_char, bool);
type CommitFilterFn = unsafe extern "C" fn(*mut c_void);
type ExecuteFilterFn = unsafe extern "C" fn(*mut c_void);
type GetDeviceErrorFn = unsafe extern "C" fn(*mut c_void, *mut *const c_char) -> i32;
type ReleaseFilterFn = unsafe extern "C" fn(*mut c_void);
type ReleaseDeviceFn = unsafe extern "C" fn(*mut c_void);

struct Oidn {
    new_device: NewDeviceFn,
    commit_device: CommitDeviceFn,
    new_filter: NewFilterFn,
    set_shared_filter_image: SetSharedFilterImageFn,
    set_filter_bool: SetFilterBoolFn,
    commit_filter: CommitFilterFn,
    execute_filter: ExecuteFilterFn,
    get_device_error: GetDeviceErrorFn,
    release_filter: ReleaseFilterFn,
    release_device: ReleaseDeviceFn,
}

unsafe impl Send for Oidn {}
unsafe impl Sync for Oidn {}

static OIDN: OnceLock<Option<Oidn>> = OnceLock::new();

fn load() -> Option<Oidn> {
    let candidates = [
        "libOpenImageDenoise.dylib",
        "/opt/homebrew/lib/libOpenImageDenoise.dylib",
        "/usr/local/lib/libOpenImageDenoise.dylib",
        "libOpenImageDenoise.so",
    ];
    let handle = candidates.iter().find_map(|p| {
        let c = CString::new(*p).ok()?;
        let h = unsafe { dlopen(c.as_ptr(), RTLD_NOW) };
        (!h.is_null()).then_some(h)
    })?;
    macro_rules! sym {
        ($name:literal) => {{
            let c = CString::new($name).ok()?;
            let p = unsafe { dlsym(handle, c.as_ptr()) };
            if p.is_null() {
                return None;
            }
            unsafe { std::mem::transmute(p) }
        }};
    }
    Some(Oidn {
        new_device: sym!("oidnNewDevice"),
        commit_device: sym!("oidnCommitDevice"),
        new_filter: sym!("oidnNewFilter"),
        set_shared_filter_image: sym!("oidnSetSharedFilterImage"),
        set_filter_bool: sym!("oidnSetFilterBool"),
        commit_filter: sym!("oidnCommitFilter"),
        execute_filter: sym!("oidnExecuteFilter"),
        get_device_error: sym!("oidnGetDeviceError"),
        release_filter: sym!("oidnReleaseFilter"),
        release_device: sym!("oidnReleaseDevice"),
    })
}

/// True when the OIDN library is present and loadable.
pub fn available() -> bool {
    OIDN.get_or_init(load).is_some()
}

fn pack(img: &Image) -> Vec<f32> {
    let mut v = Vec::with_capacity(img.len() * img.first().map(|r| r.len()).unwrap_or(0) * 3);
    for row in img {
        for p in row {
            v.push(p.x as f32);
            v.push(p.y as f32);
            v.push(p.z as f32);
        }
    }
    v
}

/// OIDN-denoise the film's diffuse layer (albedo/normal-guided), add the
/// raw specular back. None when the library is unavailable or errors.
pub fn denoise_oidn(film: &Film) -> Option<Image> {
    let oidn = OIDN.get_or_init(load).as_ref()?;
    let w = film.width();
    let h = film.height();
    if w == 0 || h == 0 {
        return None;
    }

    let mut color = pack(&film.diffuse);
    let mut albedo = pack(&film.albedo);
    let mut normal = pack(&film.normal);
    let mut output = vec![0.0f32; w * h * 3];

    unsafe {
        let device = (oidn.new_device)(0); // OIDN_DEVICE_TYPE_DEFAULT
        if device.is_null() {
            return None;
        }
        (oidn.commit_device)(device);
        let rt = CString::new("RT").ok()?;
        let filter = (oidn.new_filter)(device, rt.as_ptr());
        if filter.is_null() {
            (oidn.release_device)(device);
            return None;
        }
        let name = |n: &str| CString::new(n).unwrap();
        let img = |f: &SetSharedFilterImageFn, n: &str, buf: &mut [f32]| {
            let cn = name(n);
            f(
                filter,
                cn.as_ptr(),
                buf.as_mut_ptr() as *mut c_void,
                OIDN_FORMAT_FLOAT3,
                w,
                h,
                0,
                0,
                0,
            );
        };
        img(&oidn.set_shared_filter_image, "color", &mut color);
        img(&oidn.set_shared_filter_image, "albedo", &mut albedo);
        img(&oidn.set_shared_filter_image, "normal", &mut normal);
        img(&oidn.set_shared_filter_image, "output", &mut output);
        let hdr = name("hdr");
        (oidn.set_filter_bool)(filter, hdr.as_ptr(), true);
        (oidn.commit_filter)(filter);
        (oidn.execute_filter)(filter);

        let mut err_msg: *const c_char = std::ptr::null();
        let err = (oidn.get_device_error)(device, &mut err_msg);
        (oidn.release_filter)(filter);
        (oidn.release_device)(device);
        if err != 0 {
            if !err_msg.is_null() {
                eprintln!(
                    "warning: OIDN failed ({}); falling back to a-trous",
                    CStr::from_ptr(err_msg).to_string_lossy()
                );
            }
            return None;
        }
    }

    Some(
        (0..h)
            .map(|y| {
                (0..w)
                    .map(|x| {
                        let i = (y * w + x) * 3;
                        let s = film.specular[y][x];
                        Vec3::new(
                            output[i] as f64 + s.x,
                            output[i + 1] as f64 + s.y,
                            output[i + 2] as f64 + s.z,
                        )
                    })
                    .collect()
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    /// With the library installed (this machine), OIDN must load, run,
    /// and slash variance on a noisy flat field without shifting the mean.
    #[test]
    fn oidn_denoises_when_available() {
        if !available() {
            eprintln!("OIDN not installed; skipping");
            return;
        }
        let (w, h) = (64, 64);
        let mut state = 7u64;
        let mut rand = move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (state >> 33) as f64 / (1u64 << 31) as f64
        };
        let noisy: Image = (0..h)
            .map(|_| {
                (0..w)
                    .map(|_| {
                        let v: f64 = (0.5 + (rand() - 0.5) * 0.4).max(0.0);
                        Vec3::new(v, v, v)
                    })
                    .collect()
            })
            .collect();
        let flat = |v: f64| -> Image {
            (0..h).map(|_| (0..w).map(|_| Vec3::new(v, v, v)).collect()).collect()
        };
        let film = Film {
            beauty: noisy.clone(),
            diffuse: noisy.clone(),
            specular: flat(0.0),
            albedo: flat(0.5),
            normal: (0..h)
                .map(|_| (0..w).map(|_| Vec3::new(0.0, 0.0, 1.0)).collect())
                .collect(),
            depth: flat(5.0),
            id: flat(1.0),
            manifest: BTreeMap::new(),
        };
        let out = denoise_oidn(&film).expect("OIDN run");
        let stats = |img: &Image| {
            let vals: Vec<f64> = img.iter().flatten().map(|p| p.x).collect();
            let mean = vals.iter().sum::<f64>() / vals.len() as f64;
            let var =
                vals.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / vals.len() as f64;
            (mean, var)
        };
        let (m_in, v_in) = stats(&noisy);
        let (m_out, v_out) = stats(&out);
        assert!(v_out < v_in * 0.1, "variance {v_in:.5} -> {v_out:.5}");
        assert!((m_out - m_in).abs() < 0.03, "mean {m_in:.3} -> {m_out:.3}");
    }
}
