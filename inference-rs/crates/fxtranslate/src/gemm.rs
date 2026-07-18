//! Accelerated int8 GEMM for the shifted int8 affine (`int8shiftAlphaAll`).
//!
//! [`PreparedB`] holds a weight transformed once into a register-blocked int8
//! layout; [`PreparedB::matmul`] then runs the SIMD GEMM. It computes the same
//! result as [`crate::ops::intgemm_affine`] — the scalar kernel validated against
//! the marian oracle — so `tests/gemm_parity.rs` pins this path to that scalar
//! reference (and thus, transitively, to the oracle).
//!
//! Three implementations behind one surface (`PreparedB::{new, matmul, read_row}`
//! + `backend`), selected at compile time in this precedence:
//! 1. **`gemmology_simd`** — the vendored gemmology SIMD kernel via FFI
//!    (`src/gemmology_shim.cpp`), for native aarch64 (i8mm) / x86_64 (AVX2), when
//!    `build.rs` could build the shim.
//! 2. **wasm SIMD128** — a pure-Rust kernel (`core::arch::wasm32`, `dot_i16x8_s`),
//!    when the target is `wasm32` built with `+simd128` and gemmology is not live.
//! 3. **scalar stub** — otherwise; [`PreparedB::new`] always returns `None`, so
//!    callers use [`crate::ops::intgemm_affine`].

#[cfg(gemmology_simd)]
mod imp {
    use std::os::raw::{c_char, c_void};

    extern "C" {
        fn gemmology_prepare_b(b_transposed: *const i8, n: usize, k: usize) -> *mut c_void;
        fn gemmology_free_b(handle: *mut c_void);
        fn gemmology_multiply(
            handle: *mut c_void,
            a: *const u8,
            m: usize,
            unquant: f32,
            bias: *const f32,
            out: *mut f32,
        );
        fn gemmology_prepared_bytes() -> usize;
        fn gemmology_read_row(handle: *const c_void, id: usize, out: *mut i8);
        fn gemmology_backend_name() -> *const c_char;
        fn gemmology_gemm_threads() -> usize;
    }

    /// Cores the intra-op GEMM pool will use (feature `gemm-threads` +
    /// `FXT_GEMM_THREADS`): `1` = inert (sequential), `0` = feature not compiled.
    pub fn gemm_threads() -> usize {
        // SAFETY: reads a size from the shim; always valid.
        unsafe { gemmology_gemm_threads() }
    }

    /// Total retained bytes of prepared-B weight buffers — the persistent C++
    /// allocations gemmology holds, which dhat (Rust-heap only) cannot see. For
    /// memory accounting.
    pub fn prepared_bytes() -> usize {
        // SAFETY: reads an atomic counter in the shim; always valid.
        unsafe { gemmology_prepared_bytes() }
    }

    /// The xsimd arch the SIMD kernel was compiled for ("i8mm+neon64", "avx2", …),
    /// read from the compiled shim itself. Lets callers and CI prove which kernel
    /// is live rather than trusting it didn't silently fall back to scalar — the
    /// scalar stub below reports "scalar" instead.
    pub fn backend() -> &'static str {
        // SAFETY: the shim returns a pointer to a static, NUL-terminated string
        // (xsimd's `Arch::name()`), valid for the whole program.
        let name = unsafe { std::ffi::CStr::from_ptr(gemmology_backend_name()) };
        name.to_str().unwrap_or("unknown")
    }

    /// A weight matrix prepared once into gemmology's register-blocked int8 layout.
    ///
    /// Logically a transposed weight `[n, k]` (row-major, `w[col * k + kk]`) — the
    /// same orientation [`crate::ops::intgemm_affine`] consumes. Owns aligned
    /// native memory freed on drop. Holds a raw pointer, so it is neither `Send`
    /// nor `Sync` (used single-threaded behind the engine's `&self`).
    pub struct PreparedB {
        handle: *mut c_void,
        n: usize,
        k: usize,
    }

    impl PreparedB {
        /// Prepare a logical transposed int8 weight `[n, k]`. Returns `None` if
        /// `k` is not a multiple of the int8 SIMD register width (16 on i8mm/NEON,
        /// 32 on AVX2, 64 on AVX-512), in which case the caller uses the scalar
        /// kernel.
        ///
        /// # Panics
        /// If `b_transposed.len() != n * k`.
        pub fn new(b_transposed: &[i8], n: usize, k: usize) -> Option<PreparedB> {
            assert_eq!(b_transposed.len(), n * k, "B length must be n * k");
            // SAFETY: pointer/len are consistent with (n, k); the shim only reads
            // n*k bytes and copies them into its own buffer.
            let handle = unsafe { gemmology_prepare_b(b_transposed.as_ptr(), n, k) };
            if handle.is_null() {
                None
            } else {
                Some(PreparedB { handle, n, k })
            }
        }

        /// `out[m, n] = unquant * (A[m,k] · W[k,n]) + bias[n]`, where `a` is the
        /// shifted uint8 activation `[m, k]` (row-major) and `bias` is the prepared
        /// bias of length `n`. Returns `[m, n]` row-major.
        ///
        /// # Panics
        /// If `a.len() != m * k` or `bias.len() != n`.
        pub fn matmul(&self, a: &[u8], m: usize, unquant: f32, bias: &[f32]) -> Vec<f32> {
            let mut out = Vec::new();
            self.matmul_into(a, m, unquant, bias, &mut out);
            out
        }

        /// [`matmul`] into a caller-owned buffer (resized to `[m, n]`), reused
        /// across calls to avoid a fresh allocation per GEMM. The shim's own
        /// A/bias/output scratch is likewise persistent, so a steady-state call
        /// allocates nothing.
        ///
        /// # Panics
        /// If `a.len() != m * k` or `bias.len() != n`.
        pub fn matmul_into(
            &self,
            a: &[u8],
            m: usize,
            unquant: f32,
            bias: &[f32],
            out: &mut Vec<f32>,
        ) {
            assert_eq!(a.len(), m * self.k, "A length must be m * k");
            assert_eq!(bias.len(), self.n, "bias length must be n");
            out.clear();
            out.resize(m * self.n, 0.0);
            // SAFETY: buffers match the shim's expected dimensions; `out` is sized
            // m*n and fully written. The shim mutates only its own C++ scratch.
            unsafe {
                gemmology_multiply(
                    self.handle,
                    a.as_ptr(),
                    m,
                    unquant,
                    bias.as_ptr(),
                    out.as_mut_ptr(),
                );
            }
        }

        /// Read logical row `id` of the prepared weight back into `out` (length
        /// `k`), reversing the register-blocked pack. This lets a caller drop the
        /// raw int8 copy and serve row lookups (e.g. embeddings) out of this one
        /// packed buffer.
        ///
        /// # Panics
        /// If `out.len() != k` or `id >= n`.
        pub fn read_row(&self, id: usize, out: &mut [i8]) {
            assert_eq!(out.len(), self.k, "out length must be k");
            assert!(id < self.n, "row id {id} out of range (n = {})", self.n);
            // SAFETY: id < n and out has length k; the shim writes exactly k bytes
            // into out and only reads its own packed buffer.
            unsafe { gemmology_read_row(self.handle, id, out.as_mut_ptr()) };
        }
    }

    impl Drop for PreparedB {
        fn drop(&mut self) {
            // SAFETY: `handle` came from gemmology_prepare_b and is freed exactly once.
            unsafe { gemmology_free_b(self.handle) };
        }
    }

    // SAFETY: the packed weight buffer behind `handle` is written once by
    // `gemmology_prepare_b` and is thereafter read-only — `matmul`/`read_row` only
    // read it, and the shim allocates its A/bias/output scratch per call (see
    // `gemmology_shim.cpp`), so it holds no shared mutable state. Concurrent calls
    // on the same `&PreparedB` from multiple threads therefore only perform
    // disjoint reads, which is sound. This lets an `Engine` be shared across worker
    // threads (feature `threads`) without duplicating the packed weights. Not
    // `Send`: ownership/drop should stay on the thread that created it.
    unsafe impl Sync for PreparedB {}
}

/// Pure-Rust wasm SIMD128 int8 kernel (`core::arch::wasm32`). A third `PreparedB`
/// twin of [`crate::ops::intgemm_affine`], reached only when gemmology's FFI
/// kernel is absent and the wasm32 target was built with `+simd128`. No C++, no
/// emscripten. Because it accumulates products at full i32 precision (via
/// `i32x4.dot_i16x8_s`, no int16 intermediate), it is an *exact* backend — bit-
/// close to the scalar oracle on any input, like scalar / ARM usdot / x86 VNNI.
#[cfg(all(
    not(gemmology_simd),
    target_arch = "wasm32",
    target_feature = "simd128"
))]
mod imp {
    use core::arch::wasm32::*;

    /// Always 0 — this kernel is pure Rust with no persistent C++ allocation to
    /// account for (its packed buffer lives on the Rust heap, visible to dhat).
    pub fn prepared_bytes() -> usize {
        0
    }

    /// The wasm SIMD128 kernel is live. CI's `FXTRANSLATE_REQUIRE_SIMD` gate asserts
    /// this is not `"scalar"` (see tests/gemm_parity.rs).
    pub fn backend() -> &'static str {
        "wasm-simd128"
    }

    /// Always 1 — single-threaded; wasm threads are out of scope (notes/12).
    pub fn gemm_threads() -> usize {
        1
    }

    /// Number of int8 lanes in a 128-bit register — the block width we pack `B` in
    /// and the register-width gate (`k % 16 == 0`), matching the NEON/SSE path.
    const LANES: usize = 16;

    /// A weight matrix prepared into a SIMD-friendly int8 layout that we own.
    ///
    /// Logically the transposed weight `[n, k]` (row-major, `w[col*k + kk]`), the
    /// same orientation [`crate::ops::intgemm_affine`] consumes. The packed layout
    /// is simply the row-major `[n, k]` bytes with each row's `k` zero-padded up to
    /// a multiple of `LANES` — but since `PreparedB::new` only accepts `k % 16 == 0`
    /// there is no padding in practice, so a packed row is bit-identical to the
    /// logical row. Keeping the layout this transparent makes `read_row` a plain
    /// copy (the inverse we control, unlike gemmology's opaque tiling).
    pub struct PreparedB {
        /// Packed int8 weight, `n` rows of `k_padded` bytes each.
        packed: Vec<i8>,
        n: usize,
        k: usize,
        k_padded: usize,
    }

    impl PreparedB {
        /// Prepare a logical transposed int8 weight `[n, k]`. Returns `None` unless
        /// `k` is a multiple of the 16-lane int8 register width (so the caller keeps
        /// the scalar kernel). The transformer's `k = 384/1536` qualify.
        ///
        /// # Panics
        /// If `b_transposed.len() != n * k`.
        pub fn new(b_transposed: &[i8], n: usize, k: usize) -> Option<PreparedB> {
            assert_eq!(b_transposed.len(), n * k, "B length must be n * k");
            if k % LANES != 0 {
                return None;
            }
            // No padding needed once `k % 16 == 0`, so the pack is the identity copy;
            // the field is kept general so the layout note above holds if the gate
            // is ever relaxed.
            let k_padded = k;
            Some(PreparedB {
                packed: b_transposed.to_vec(),
                n,
                k,
                k_padded,
            })
        }

        /// `out[m, n] = unquant * (A[m,k] · W[k,n]) + bias[n]`, where `a` is the
        /// shifted uint8 activation `[m, k]` (row-major) and `bias` is the prepared
        /// bias of length `n`. Returns `[m, n]` row-major.
        ///
        /// # Panics
        /// If `a.len() != m * k` or `bias.len() != n`.
        pub fn matmul(&self, a: &[u8], m: usize, unquant: f32, bias: &[f32]) -> Vec<f32> {
            let mut out = Vec::new();
            self.matmul_into(a, m, unquant, bias, &mut out);
            out
        }

        /// [`matmul`] into a caller-owned buffer (resized to `[m, n]`), reused across
        /// calls to avoid a fresh allocation per GEMM.
        ///
        /// # Panics
        /// If `a.len() != m * k` or `bias.len() != n`.
        pub fn matmul_into(
            &self,
            a: &[u8],
            m: usize,
            unquant: f32,
            bias: &[f32],
            out: &mut Vec<f32>,
        ) {
            assert_eq!(a.len(), m * self.k, "A length must be m * k");
            assert_eq!(bias.len(), self.n, "bias length must be n");
            out.clear();
            out.resize(m * self.n, 0.0);
            // SAFETY: this build enabled `simd128` (the module cfg), so the wasm SIMD
            // instructions in `dot` are available. Slices are bounds-checked here and
            // `dot` reads exactly `k` bytes from each.
            for row in 0..m {
                let a_row = &a[row * self.k..(row + 1) * self.k];
                for col in 0..self.n {
                    let b_col = &self.packed[col * self.k_padded..col * self.k_padded + self.k];
                    let acc = unsafe { dot(a_row, b_col) };
                    out[row * self.n + col] = unquant * acc as f32 + bias[col];
                }
            }
        }

        /// Read logical row `id` of the prepared weight back into `out` (length `k`).
        /// Our pack is the identity copy (no tiling), so this is a plain slice copy —
        /// the inverse of `new`. Lets `lean-embed` serve embedding rows out of the
        /// one packed buffer.
        ///
        /// # Panics
        /// If `out.len() != k` or `id >= n`.
        pub fn read_row(&self, id: usize, out: &mut [i8]) {
            assert_eq!(out.len(), self.k, "out length must be k");
            assert!(id < self.n, "row id {id} out of range (n = {})", self.n);
            let base = id * self.k_padded;
            out.copy_from_slice(&self.packed[base..base + self.k]);
        }
    }

    /// Exact int8 dot product of a `u8` activation row and an `i8` weight column,
    /// both length `k` (a multiple of 16). Widens each 16-byte block to two
    /// `i16x8` halves — `a` zero-extended (it is unsigned, range 0..=254), `b`
    /// sign-extended — then multiplies-and-pairwise-adds with `i32x4.dot_i16x8_s`,
    /// whose four i32 lanes each hold `a0*b0 + a1*b1` formed at i32 precision. There
    /// is no int16 accumulator to saturate (unlike x86 `maddubs`), so the result is
    /// exactly `Σ a[i]*b[i]` — matching the scalar reference bit-for-bit.
    ///
    /// # Safety
    /// Requires the `simd128` target feature (guaranteed by the module cfg). `a`
    /// and `b` must both have length a multiple of 16.
    #[target_feature(enable = "simd128")]
    unsafe fn dot(a: &[u8], b: &[i8]) -> i32 {
        debug_assert_eq!(a.len(), b.len());
        debug_assert_eq!(a.len() % LANES, 0);
        let mut acc = i32x4_splat(0);
        let mut off = 0;
        while off < a.len() {
            // Load 16 bytes of each operand.
            let av = v128_load(a.as_ptr().add(off) as *const v128);
            let bv = v128_load(b.as_ptr().add(off) as *const v128);
            // Widen to i16x8: A is unsigned so zero-extend (values land in 0..=254,
            // well within the non-negative i16 range); B is signed so sign-extend.
            let a_lo = i16x8_extend_low_u8x16(av);
            let a_hi = i16x8_extend_high_u8x16(av);
            let b_lo = i16x8_extend_low_i8x16(bv);
            let b_hi = i16x8_extend_high_i8x16(bv);
            // Pairwise multiply-add straight into i32 lanes — no int16 intermediate,
            // so nothing to saturate; the accumulation is exact.
            acc = i32x4_add(acc, i32x4_dot_i16x8(a_lo, b_lo));
            acc = i32x4_add(acc, i32x4_dot_i16x8(a_hi, b_hi));
            off += LANES;
        }
        // Horizontal sum of the four i32 lanes.
        i32x4_extract_lane::<0>(acc)
            + i32x4_extract_lane::<1>(acc)
            + i32x4_extract_lane::<2>(acc)
            + i32x4_extract_lane::<3>(acc)
    }
}

/// Scalar-fallback stub: no accelerated kernel is live for this target (no wired
/// gemmology arch — i.e. not aarch64 or x86_64 — nor wasm32+simd128, or `portable`,
/// or no C++ compiler). [`PreparedB::new`] always returns `None`, so every caller
/// keeps the scalar [`crate::ops::intgemm_affine`] path; the compute methods exist
/// only to satisfy those call sites and are never reached.
#[cfg(all(
    not(gemmology_simd),
    not(all(target_arch = "wasm32", target_feature = "simd128"))
))]
mod imp {
    /// Always 0 — no SIMD weights are prepared without a kernel.
    pub fn prepared_bytes() -> usize {
        0
    }

    /// "scalar" — no SIMD kernel was compiled, so every GEMM uses
    /// [`crate::ops::intgemm_affine`]. A caller/CI check that requires a real SIMD
    /// backend asserts against this value (see tests/gemm_parity.rs).
    pub fn backend() -> &'static str {
        "scalar"
    }

    /// Always 0 — no SIMD kernel, so no intra-op GEMM pool.
    pub fn gemm_threads() -> usize {
        0
    }

    /// Stub mirror of the SIMD [`PreparedB`]; never constructed.
    pub struct PreparedB {
        _never: (),
    }

    impl PreparedB {
        /// Always `None` on a scalar build — the caller uses the scalar kernel.
        pub fn new(b_transposed: &[i8], n: usize, k: usize) -> Option<PreparedB> {
            debug_assert_eq!(b_transposed.len(), n * k, "B length must be n * k");
            None
        }

        pub fn matmul(&self, _a: &[u8], _m: usize, _unquant: f32, _bias: &[f32]) -> Vec<f32> {
            unreachable!("scalar-fallback PreparedB is never constructed")
        }

        pub fn matmul_into(
            &self,
            _a: &[u8],
            _m: usize,
            _unquant: f32,
            _bias: &[f32],
            _out: &mut Vec<f32>,
        ) {
            unreachable!("scalar-fallback PreparedB is never constructed")
        }

        pub fn read_row(&self, _id: usize, _out: &mut [i8]) {
            unreachable!("scalar-fallback PreparedB is never constructed")
        }
    }
}

pub use imp::{backend, gemm_threads, prepared_bytes, PreparedB};
