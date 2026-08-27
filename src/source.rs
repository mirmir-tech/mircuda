use std::borrow::Cow;

/// CUDA source embedded in the Rust binary for NVRTC compilation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KernelSource {
    name: &'static str,
    body: KernelSourceBody,
}

/// Precompiled PTX embedded in the binary for one exact CUDA compute capability.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PtxSource {
    name: &'static str,
    body: &'static str,
    compute_capability: (i32, i32),
}

impl PtxSource {
    /// Declares an embedded PTX module and the exact device capability it targets.
    #[must_use]
    pub const fn embedded(
        name: &'static str,
        body: &'static str,
        compute_capability: (i32, i32),
    ) -> Self {
        Self { name, body, compute_capability }
    }

    /// Diagnostic module name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        self.name
    }

    /// PTX text passed directly to the CUDA driver.
    #[must_use]
    pub const fn body(self) -> &'static str {
        self.body
    }

    /// Exact device compute capability required by the generated module.
    #[must_use]
    pub const fn compute_capability(self) -> (i32, i32) {
        self.compute_capability
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum KernelSourceBody {
    Single(&'static str),
    Parts(&'static [&'static str]),
}

impl KernelSource {
    /// Creates an inline CUDA translation unit validated by [`crate::cuda_kernel`].
    #[must_use]
    pub const fn inline(source: &'static str) -> Self {
        Self {
            name: "inline.cu",
            body: KernelSourceBody::Single(source),
        }
    }

    /// Creates a translation unit embedded with [`crate::cuda_kernel_file`].
    #[must_use]
    pub const fn embedded(name: &'static str, source: &'static str) -> Self {
        Self {
            name,
            body: KernelSourceBody::Single(source),
        }
    }

    /// Creates one translation unit from ordered, embedded source fragments.
    #[must_use]
    pub const fn composed(name: &'static str, parts: &'static [&'static str]) -> Self {
        Self {
            name,
            body: KernelSourceBody::Parts(parts),
        }
    }

    /// Returns the diagnostic name passed to the compiler.
    #[must_use]
    pub const fn name(self) -> &'static str {
        self.name
    }

    /// Returns the complete CUDA source.
    #[must_use]
    pub fn source(self) -> Cow<'static, str> {
        match self.body {
            KernelSourceBody::Single(source) => Cow::Borrowed(source),
            KernelSourceBody::Parts(parts) => Cow::Owned(parts.concat()),
        }
    }
}
